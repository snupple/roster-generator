#!/usr/bin/env python3
"""Local desktop GUI for the Roster Generator using PySimpleGUI.

Features:
- Load people.csv, shifts.csv, optional prefill.csv
- Add per-person unavailability interactively (session-scoped)
- Run greedy or PuLP optimizer (PuLP optional)
- View roster in a table and save CSV/XLSX

Run: python gui.py
"""
import io
import csv
import tempfile
import os
import sys
from collections import defaultdict, namedtuple
from datetime import datetime

try:
    import PySimpleGUI as sg
except Exception:
    print("PySimpleGUI is required. Install with: pip install PySimpleGUI")
    sys.exit(1)

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

try:
    import pulp
    HAS_PULP = True
except Exception:
    HAS_PULP = False

Person = namedtuple("Person", ["name", "max_shifts", "max_per_day", "skills", "available", "unavailable"])
Shift = namedtuple("Shift", ["sid", "day", "slot", "required", "required_skills", "start", "end"])

# Utilities copied/adapted from the Flask app

def parse_semicolon_list(s):
    if s is None:
        return set()
    return set([t.strip() for t in str(s).split(";") if t.strip()])


def parse_time_or_none(val):
    if not val or (isinstance(val, float) and str(val) == "nan"):
        return None
    try:
        if ":" in str(val):
            return datetime.strptime(str(val).strip(), "%H:%M").time()
        else:
            h = int(val)
            return datetime.strptime(f"{h:02d}:00", "%H:%M").time()
    except Exception:
        return None


def read_people_file_path(path):
    people = {}
    if HAS_PANDAS:
        df = pd.read_csv(path).fillna("")
        for _, row in df.iterrows():
            name = row.get("name") or row.get("Name")
            if not name:
                continue
            max_shifts = int(row.get("max_shifts") or row.get("maxShifts") or 999)
            max_per_day = int(row.get("max_per_day") or row.get("maxPerDay") or 999)
            skills = parse_semicolon_list(row.get("skills") or "")
            available = parse_semicolon_list(row.get("available_slots") or row.get("available") or "")
            unavailable = parse_semicolon_list(row.get("unavailable_slots") or row.get("unavailable") or "")
            people[str(name)] = Person(name=str(name), max_shifts=max_shifts, max_per_day=max_per_day,
                                       skills=skills, available=available, unavailable=unavailable)
    else:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("name") or row.get("Name")
                if not name:
                    continue
                max_shifts = int(row.get("max_shifts") or row.get("maxShifts") or 999)
                max_per_day = int(row.get("max_per_day") or 999)
                skills = parse_semicolon_list(row.get("skills") or "")
                available = parse_semicolon_list(row.get("available_slots") or row.get("available") or "")
                unavailable = parse_semicolon_list(row.get("unavailable_slots") or row.get("unavailable") or "")
                people[name] = Person(name=name, max_shifts=max_shifts, max_per_day=max_per_day,
                                      skills=skills, available=available, unavailable=unavailable)
    return people


def read_shifts_file_path(path):
    shifts = []
    if HAS_PANDAS:
        df = pd.read_csv(path).fillna("")
        for _, row in df.iterrows():
            day = row.get("day") or row.get("Day")
            slot = row.get("slot") or row.get("Slot")
            sid = row.get("id") or f"{day}_{slot}"
            required = int(row.get("required") or 1)
            req_sk = parse_semicolon_list(row.get("required_skills") or "")
            start = parse_time_or_none(row.get("start_time") or row.get("start_hour") or "")
            end = parse_time_or_none(row.get("end_time") or row.get("end_hour") or "")
            shifts.append(Shift(sid=str(sid), day=str(day), slot=str(slot), required=required,
                                required_skills=req_sk, start=start, end=end))
    else:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                day = row.get("day") or row.get("Day")
                slot = row.get("slot") or row.get("Slot")
                sid = row.get("id") or f"{day}_{slot}"
                required = int(row.get("required") or 1)
                req_sk = parse_semicolon_list(row.get("required_skills") or "")
                start = parse_time_or_none(row.get("start_time") or row.get("start_hour") or "")
                end = parse_time_or_none(row.get("end_time") or row.get("end_hour") or "")
                shifts.append(Shift(sid=str(sid), day=str(day), slot=str(slot), required=required,
                                    required_skills=req_sk, start=start, end=end))
    return shifts


def shifts_overlap(s1: Shift, s2: Shift, min_rest_hours=0):
    # handle overnight shifts
    if not s1.day or not s2.day:
        return min_rest_hours > 0 and s1.day == s2.day
    if s1.start and s1.end and s2.start and s2.end:
        def time_seconds(t):
            return t.hour*3600 + t.minute*60 + (t.second if hasattr(t, 'second') else 0)
        di_map = { 'mon':0,'tue':1,'wed':2,'thu':3,'fri':4,'sat':5,'sun':6 }
        def di(day):
            return di_map.get(str(day).strip().lower())
        di1 = di(s1.day); di2 = di(s2.day)
        if di1 is None or di2 is None:
            return min_rest_hours > 0 and s1.day == s2.day
        b1s = di1*86400 + time_seconds(s1.start)
        b1e = di1*86400 + time_seconds(s1.end)
        if b1e <= b1s: b1e += 86400
        b2s = di2*86400 + time_seconds(s2.start)
        b2e = di2*86400 + time_seconds(s2.end)
        if b2e <= b2s: b2e += 86400
        for shift in (-86400,0,86400):
            s2s = b2s + shift; s2e = b2e + shift
            if not (b1e <= s2s or s2e <= b1s):
                return True
            gap = (s2s - b1e)/3600.0
            if gap < min_rest_hours and gap >= -1e-6:
                return True
            gap2 = (b1s - s2e)/3600.0
            if gap2 < min_rest_hours and gap2 >= -1e-6:
                return True
        return False
    return min_rest_hours > 0 and s1.day == s2.day


def greedy_assign(people, shifts, prefilled, min_rest_hours=8):
    people_state = {n:{"assigned":[],"count":0} for n in people}
    roster = {}
    for s in shifts:
        assigned = []
        if s.sid in prefilled:
            for name in prefilled[s.sid]:
                if name in people:
                    assigned.append(name)
                    people_state[name]["assigned"].append(s.sid)
                    people_state[name]["count"] += 1
        roster[s.sid] = {"shift":s, "assigned":assigned}
    sorted_shifts = sorted(shifts, key=lambda x:(x.day or "", x.slot or ""))
    for s in sorted_shifts:
        need = s.required - len(roster[s.sid]["assigned"])
        for _ in range(max(0, need)):
            elig = []
            for name, p in people.items():
                st = people_state[name]
                if s.sid not in p.available:
                    continue
                if s.sid in p.unavailable:
                    continue
                if st["count"] >= p.max_shifts:
                    continue
                assigned_today = sum(1 for sid in st["assigned"] if sid.startswith(f"{s.day}_"))
                if assigned_today >= p.max_per_day:
                    continue
                if s.required_skills and not (s.required_skills <= p.skills):
                    continue
                elig.append((st["count", name]))
            if not elig:
                roster[s.sid]["assigned"].append("UNFILLED")
                continue
            elig.sort(key=lambda x:(x[0], x[1]))
            chosen = elig[0][1]
            roster[s.sid]["assigned"].append(chosen)
            people_state[chosen]["assigned"].append(s.sid)
            people_state[chosen]["count"] += 1
    return roster


def optimize_assign(people, shifts, prefilled, min_rest_hours=8, time_limit=30):
    if not HAS_PULP:
        raise RuntimeError("PuLP not installed")
    persons = list(people.keys())
    sids = [s.sid for s in shifts]
    prob = pulp.LpProblem("roster", pulp.LpMinimize)
    x = {}
    for p in persons:
        for s in sids:
            x[(p,s)] = pulp.LpVariable(f"x_{p}_{s}", cat="Binary")
    load = {p: pulp.lpSum([x[(p,s)] for s in sids]) for p in persons}
    maxLoad = pulp.LpVariable("maxLoad", lowBound=0, cat="Integer")
    minLoad = pulp.LpVariable("minLoad", lowBound=0, cat="Integer")
    prob += maxLoad - minLoad
    for s in shifts:
        pre = prefilled.get(s.sid, [])
        for name in pre:
            if name in people:
                prob += x[(name, s.sid)] == 1
        prob += pulp.lpSum([x[(p, s.sid)] for p in persons]) == s.required
    for p_name, p in people.items():
        prob += load[p_name] <= p.max_shifts
        prob += maxLoad >= load[p_name]
        prob += minLoad <= load[p_name]
        days = defaultdict(list)
        for sh in shifts:
            days[sh.day].append(sh.sid)
        for day, sidlist in days.items():
            prob += pulp.lpSum([x[(p_name, s)] for s in sidlist]) <= p.max_per_day
        for sh in shifts:
            if sh.sid not in p.available:
                prob += x[(p_name, sh.sid)] == 0
            if sh.required_skills and not (sh.required_skills <= p.skills):
                prob += x[(p_name, sh.sid)] == 0
            if sh.sid in p.unavailable:
                prob += x[(p_name, sh.sid)] == 0
    for p in persons:
        for i in range(len(shifts)):
            for j in range(i+1, len(shifts)):
                s1 = shifts[i]; s2 = shifts[j]
                if shifts_overlap(s1, s2, min_rest_hours):
                    prob += x[(p, s1.sid)] + x[(p, s2.sid)] <= 1
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    prob.solve(solver)
    roster = {}
    for s in shifts:
        assigned = []
        for p in persons:
            val = pulp.value(x[(p, s.sid)])
            if val is not None and val > 0.5:
                assigned.append(p)
        if not assigned:
            assigned = ["UNFILLED"]
        roster[s.sid] = {"shift": s, "assigned": assigned}
    return roster


# GUI state
unavailability = {}  # person -> [slot, ...]
current_people = {}
current_shifts = []
current_prefill = {}
current_roster = {}

# GUI layout
file_col = [
    [sg.Text('People CSV'), sg.Input(key='-PEOPLE-', enable_events=True, readonly=True), sg.FileBrowse(file_types=(('CSV Files','*.csv'),))],
    [sg.Text('Shifts CSV'), sg.Input(key='-SHIFTS-', enable_events=True, readonly=True), sg.FileBrowse(file_types=(('CSV Files','*.csv'),))],
    [sg.Text('Prefill CSV (optional)'), sg.Input(key='-PREFILL-', enable_events=True, readonly=True), sg.FileBrowse(file_types=(('CSV Files','*.csv'),))],
    [sg.Checkbox('Use optimizer (PuLP)', key='-OPT-')],
    [sg.Text('Minimum rest hours'), sg.Input('8', size=(6,1), key='-MINREST-')],
    [sg.Button('Add Unavailability', key='-ADDUN-'), sg.Button('View Unavailability', key='-VIEWUN-')],
    [sg.Button('Run', key='-RUN-'), sg.Button('Save CSV', key='-SAVECSV-'), sg.Button('Save XLSX', key='-SAVEXLSX-'), sg.Button('Exit')]
]

table_col = [
    [sg.Text('Roster results')],
    [sg.Table(values=[], headings=['Day','Slot','ID','Assigned'], key='-TABLE-', auto_size_columns=False, col_widths=[10,12,18,40], num_rows=20, enable_events=False)]
]

layout = [[sg.Column(file_col), sg.VerticalSeparator(), sg.Column(table_col)]]

window = sg.Window('Roster Generator — Local', layout, finalize=True)


def load_inputs():
    global current_people, current_shifts, current_prefill
    people_path = values.get('-PEOPLE-')
    shifts_path = values.get('-SHIFTS-')
    prefill_path = values.get('-PREFILL-')
    if people_path and os.path.exists(people_path):
        current_people = read_people_file_path(people_path)
    if shifts_path and os.path.exists(shifts_path):
        current_shifts = read_shifts_file_path(shifts_path)
    if prefill_path and os.path.exists(prefill_path):
        # read prefill
        current_prefill = defaultdict(list)
        if HAS_PANDAS:
            df = pd.read_csv(prefill_path).fillna("")
            for _, row in df.iterrows():
                day = row.get('day') or row.get('Day')
                slot = row.get('slot') or row.get('Slot')
                sid = row.get('id') or f"{day}_{slot}"
                assigned_raw = str(row.get('assigned') or '')
                for n in [x.strip() for x in assigned_raw.split(';') if x.strip()]:
                    current_prefill[str(sid)].append(n)
        else:
            with open(prefill_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    day = row.get('day') or row.get('Day')
                    slot = row.get('slot') or row.get('Slot')
                    sid = row.get('id') or f"{day}_{slot}"
                    assigned_raw = row.get('assigned') or ''
                    for n in [x.strip() for x in assigned_raw.split(';') if x.strip()]:
                        current_prefill[str(sid)].append(n)


def run_assignment(use_optimizer, min_rest):
    global current_roster
    if not current_people or not current_shifts:
        sg.popup('Please load People and Shifts CSV files first', title='Missing inputs')
        return
    # merge session unavailability
    people = {n: Person(name=p.name, max_shifts=p.max_shifts, max_per_day=p.max_per_day, skills=p.skills, available=p.available, unavailable=set(p.unavailable) | set(unavailability.get(n, []))) for n,p in current_people.items()}
    try:
        if use_optimizer and HAS_PULP:
            roster = optimize_assign(people, current_shifts, current_prefill, min_rest_hours=min_rest)
        else:
            roster = greedy_assign(people, current_shifts, current_prefill, min_rest_hours=min_rest)
    except Exception as e:
        sg.popup(f'Assignment failed: {e}', title='Error')
        return
    current_roster = roster
    # prepare table rows
    rows = []
    for sid, entry in roster.items():
        s = entry['shift']
        rows.append([s.day, s.slot, s.sid, ';'.join(entry['assigned'])])
    window['-TABLE-'].update(values=rows)
    sg.popup('Assignment complete', title='Done')


def save_csv(path):
    if not current_roster:
        sg.popup('No roster to save', title='Nothing')
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['day','slot','id','assigned'])
        writer.writeheader()
        for sid, entry in current_roster.items():
            s = entry['shift']
            writer.writerow({'day': s.day, 'slot': s.slot, 'id': s.sid, 'assigned': ';'.join(entry['assigned'])})
    sg.popup('CSV saved', title='Saved')


def save_xlsx(path):
    if not HAS_PANDAS:
        sg.popup('pandas required for xlsx export', title='Missing dependency')
        return
    if not current_roster:
        sg.popup('No roster to save', title='Nothing')
        return
    rows = []
    for sid, entry in current_roster.items():
        s = entry['shift']
        rows.append({'day': s.day, 'slot': s.slot, 'id': s.sid, 'assigned': ';'.join(entry['assigned'])})
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Roster_List', index=False)
    sg.popup('XLSX saved', title='Saved')

# Main event loop
while True:
    event, values = window.read()
    if event in (sg.WINDOW_CLOSED, 'Exit'):
        break
    if event == '-PEOPLE-' or event == '-SHIFTS-' or event == '-PREFILL-':
        # don't need to do anything immediate
        pass
    if event == '-ADDUN-':
        # need people loaded
        if not current_people:
            sg.popup('Load people CSV first', title='No people')
            continue
        person_list = sorted(current_people.keys())
        layout_un = [
            [sg.Text('Person'), sg.Combo(person_list, key='-U_PERSON-')],
            [sg.Text('Unavailable slots (semicolon-separated)'), sg.Input(key='-U_SLOTS-')],
            [sg.Button('Save'), sg.Button('Cancel')]
        ]
        win_un = sg.Window('Add Unavailability', layout_un, modal=True)
        e, v = win_un.read()
        if e == 'Save':
            p = v['-U_PERSON-']
            slots = v['-U_SLOTS-'] or ''
            slotlist = [s.strip() for s in slots.split(';') if s.strip()]
            if p:
                unavailability[p] = slotlist
                sg.popup(f'Saved unavailability for {p}', title='Saved')
        win_un.close()
    if event == '-VIEWUN-':
        if not unavailability:
            sg.popup('No unavailability set in this session', title='None')
        else:
            text = '\n'.join([f"{p}: {';'.join(slots)}" for p, slots in unavailability.items()])
            sg.popup_scrolled(text, title='Unavailability')
    if event == '-RUN-':
        load_inputs()
        use_opt = values.get('-OPT-')
        try:
            min_rest = float(values.get('-MINREST-') or 8.0)
        except Exception:
            min_rest = 8.0
        run_assignment(use_opt, min_rest)
    if event == '-SAVECSV-':
        path = sg.popup_get_file('Save CSV', save_as=True, file_types=(('CSV Files','*.csv'),), default_extension='csv')
        if path:
            save_csv(path)
    if event == '-SAVEXLSX-':
        path = sg.popup_get_file('Save XLSX', save_as=True, file_types=(('Excel Files','*.xlsx'),), default_extension='xlsx')
        if path:
            save_xlsx(path)

window.close()