#!/usr/bin/env python3
"""
Flask web UI for the Roster Generator (AJAX version)

Run:
    export FLASK_APP=app.py
    flask run

This version exposes /api/run which returns JSON and tokens for downloads.
"""
import io
import csv
import tempfile
import uuid
import os
import threading
from collections import defaultdict, namedtuple
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for

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

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET", "change-me")

Person = namedtuple("Person", ["name", "max_shifts", "max_per_day", "skills", "available", "unavailable"])
Shift = namedtuple("Shift", ["sid", "day", "slot", "required", "required_skills", "start", "end"])

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_data")
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Token -> { path, type, filename }
DOWNLOAD_STORE = {}
DOWNLOAD_LOCK = threading.Lock()


def _cleanup_token_later(token, delay_seconds=3600):
    """Schedule cleanup of a token after delay_seconds (best-effort background cleanup)."""
    def _cleanup():
        import time
        time.sleep(delay_seconds)
        with DOWNLOAD_LOCK:
            item = DOWNLOAD_STORE.pop(token, None)
        if item:
            try:
                os.remove(item["path"])
            except Exception:
                pass
    t = threading.Thread(target=_cleanup, daemon=True)
    t.start()


def parse_semicolon_list(s):
    if s is None:
        return set()
    return set([t.strip() for t in str(s).split(";") if t.strip()])


def parse_time_or_none(val):
    if not val:
        return None
    try:
        if ":" in str(val):
            return datetime.strptime(str(val).strip(), "%H:%M").time()
        else:
            h = int(val)
            return datetime.strptime(f"{h:02d}:00", "%H:%M").time()
    except Exception:
        return None


def read_people_file(fobj):
    """Read people from a file-like object (binary)."""
    data = fobj.read()
    if isinstance(data, bytes):
        text = data.decode("utf-8")
    else:
        text = str(data)
    if HAS_PANDAS:
        df = pd.read_csv(io.StringIO(text)).fillna("")
        people = {}
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
        return people
    else:
        reader = csv.DictReader(io.StringIO(text))
        people = {}
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


def read_shifts_file(fobj):
    data = fobj.read()
    if isinstance(data, bytes):
        text = data.decode("utf-8")
    else:
        text = str(data)
    if HAS_PANDAS:
        df = pd.read_csv(io.StringIO(text)).fillna("")
        shifts = []
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
        return shifts
    else:
        reader = csv.DictReader(io.StringIO(text))
        shifts = []
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


def read_prefill_file(fobj):
    if fobj is None:
        return {}
    data = fobj.read()
    if isinstance(data, bytes):
        text = data.decode("utf-8")
    else:
        text = str(data)
    if HAS_PANDAS:
        df = pd.read_csv(io.StringIO(text)).fillna("")
        prefilled = defaultdict(list)
        for _, row in df.iterrows():
            day = row.get("day") or row.get("Day")
            slot = row.get("slot") or row.get("Slot")
            sid = row.get("id") or f"{day}_{slot}"
            assigned_raw = str(row.get("assigned") or "")
            names = [n.strip() for n in assigned_raw.split(";") if n.strip()]
            prefilled[str(sid)].extend(names)
        return prefilled
    else:
        reader = csv.DictReader(io.StringIO(text))
        prefilled = defaultdict(list)
        for row in reader:
            day = row.get("day") or row.get("Day")
            slot = row.get("slot") or row.get("Slot")
            sid = row.get("id") or f"{day}_{slot}"
            assigned_raw = row.get("assigned") or ""
            names = [n.strip() for n in assigned_raw.split(";") if n.strip()]
            prefilled[sid].extend(names)
        return prefilled


def shifts_overlap(s1: Shift, s2: Shift, min_rest_hours=0):
    if s1.day != s2.day:
        return False
    if s1.start and s1.end and s2.start and s2.end:
        dt1_end = datetime.combine(datetime.min, s1.end)
        dt2_start = datetime.combine(datetime.min, s2.start)
        gap = (dt2_start - dt1_end).total_seconds() / 3600.0
        if gap < min_rest_hours and gap >= -0.001:
            return True
        dt2_end = datetime.combine(datetime.min, s2.end)
        dt1_start = datetime.combine(datetime.min, s1.start)
        gap2 = (dt1_start - dt2_end).total_seconds() / 3600.0
        if gap2 < min_rest_hours and gap2 >= -0.001:
            return True
        if not (dt1_end <= dt2_start or dt2_end <= dt1_start):
            return True
        return False
    return min_rest_hours > 0 and s1.day == s2.day


def greedy_assign(people, shifts, prefilled, min_rest_hours=8):
    people_state = {n: {"assigned": [], "count": 0} for n in people}
    roster = {}
    for s in shifts:
        assigned = []
        if s.sid in prefilled:
            for name in prefilled[s.sid]:
                if name in people:
                    assigned.append(name)
                    people_state[name]["assigned"].append(s.sid)
                    people_state[name]["count"] += 1
        roster[s.sid] = {"shift": s, "assigned": assigned}
    sorted_shifts = sorted(shifts, key=lambda x: (x.day or "", x.slot or ""))
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
                elig.append((st["count"], name))
            if not elig:
                roster[s.sid]["assigned"].append("UNFILLED")
                continue
            elig.sort(key=lambda x: (x[0], x[1]))
            chosen = elig[0][1]
            roster[s.sid]["assigned"].append(chosen)
            people_state[chosen]["assigned"].append(s.sid)
            people_state[chosen]["count"] += 1
    return roster


def optimize_assign(people, shifts, prefilled, min_rest_hours=8, time_limit=30):
    if not HAS_PULP:
        raise RuntimeError("PuLP is required for optimization. Install with `pip install pulp`.")
    persons = list(people.keys())
    sids = [s.sid for s in shifts]
    prob = pulp.LpProblem("roster", pulp.LpMinimize)
    x = {}
    for p in persons:
        for s in sids:
            x[(p, s)] = pulp.LpVariable(f"x_{p}_{s}", cat="Binary")
    load = {p: pulp.lpSum([x[(p, s)] for s in sids]) for p in persons}
    maxLoad = pulp.LpVariable("maxLoad", lowBound=0, cat="Integer")
    minLoad = pulp.Lp.Variable("minLoad", lowBound=0, cat="Integer")
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
            for j in range(i + 1, len(shifts)):
                s1 = shifts[i]
                s2 = shifts[j]
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


def roster_to_csv_bytes(roster):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["day", "slot", "id", "assigned"])
    writer.writeheader()
    for sid, entry in roster.items():
        s = entry["shift"]
        writer.writerow({"day": s.day, "slot": s.slot, "id": s.sid, "assigned": ";".join(entry["assigned"])})
    return output.getvalue().encode("utf-8")


def roster_to_xlsx_bytes(roster):
    if not HAS_PANDAS:
        raise RuntimeError("pandas required for xlsx export")
    rows = []
    for sid, entry in roster.items():
        s = entry["shift"]
        rows.append({"day": s.day, "slot": s.slot, "id": s.sid, "assigned": ";".join(entry["assigned"])})
    df = pd.DataFrame(rows)
    cal = df.pivot(index="slot", columns="day", values="assigned")
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Roster_List", index=False)
        cal.to_excel(writer, sheet_name="Roster_Calendar")
    bio.seek(0)
    return bio.read()


@app.route("/", methods=["GET"])
def index():
    samples = [f for f in os.listdir(SAMPLE_DIR) if f.endswith(".csv")]
    return render_template("index.html", samples=samples)


@app.route("/api/run", methods=["POST"])
def api_run():
    """
    Accepts multipart/form-data:
      - people_file (file)
      - shifts_file (file)
      - prefill_file (optional)
      - sample_choice (optional)
      - use_optimizer ("on" or missing)
      - min_rest (float)
    Returns JSON:
      { status: "ok", rows: [{day,slot,id,assigned}, ...], csv_token: "...", xlsx_token: "..." }
    """
    sample_choice = request.form.get("sample_choice")
    use_optimizer = bool(request.form.get("use_optimizer"))
    try:
        min_rest = float(request.form.get("min_rest") or 8.0)
    except Exception:
        min_rest = 8.0

    try:
        if sample_choice:
            # load sample files from SAMPLE_DIR
            people_path = os.path.join(SAMPLE_DIR, sample_choice)
            # try to infer a paired shifts file
            shifts_sample = sample_choice.replace("people", "shifts") if "people" in sample_choice else "sample_shifts.csv"
            shifts_path = os.path.join(SAMPLE_DIR, shifts_sample)
            prefill_path = os.path.join(SAMPLE_DIR, "sample_prefill.csv")
            with open(people_path, "rb") as pf:
                people = read_people_file(pf)
            with open(shifts_path, "rb") as sf:
                shifts = read_shifts_file(sf)
            try:
                with open(prefill_path, "rb") as prf:
                    prefilled = read_prefill_file(prf)
            except Exception:
                prefilled = {}
        else:
            people_file = request.files.get("people_file")
            shifts_file = request.files.get("shifts_file")
            prefill_file = request.files.get("prefill_file")
            if not people_file or not shifts_file:
                return jsonify({"status": "error", "message": "Please upload people and shifts CSV files"}), 400
            people = read_people_file(people_file.stream if hasattr(people_file, "stream") else people_file)
            shifts = read_shifts_file(shifts_file.stream if hasattr(shifts_file, "stream") else shifts_file)
            prefilled = read_prefill_file(prefill_file.stream if prefill_file and hasattr(prefill_file, "stream") else prefill_file) if prefill_file else {}
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to read inputs: {e}"}), 400

    # run assignment
    try:
        if use_optimizer:
            if not HAS_PULP:
                roster = greedy_assign(people, shifts, prefilled, min_rest_hours=min_rest)
            else:
                roster = optimize_assign(people, shifts, prefilled, min_rest_hours=min_rest)
        else:
            roster = greedy_assign(people, shifts, prefilled, min_rest_hours=min_rest)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Assignment failed: {e}"}), 500

    # convert roster to rows for JSON
    rows = []
    for sid, entry in roster.items():
        s = entry["shift"]
        rows.append({"day": s.day, "slot": s.slot, "id": s.sid, "assigned": ";".join(entry["assigned"])})

    # write temp CSV and optionally XLSX and create tokens
    csv_bytes = roster_to_csv_bytes(roster)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.write(csv_bytes)
    tmp.flush()
    tmp.close()
    csv_token = uuid.uuid4().hex
    with DOWNLOAD_LOCK:
        DOWNLOAD_STORE[csv_token] = {"path": tmp.name, "type": "csv", "filename": "roster_output.csv"}
    _cleanup_token_later(csv_token, delay_seconds=3600)

    xlsx_token = None
    if HAS_PANDAS:
        try:
            xlsx_bytes = roster_to_xlsx_bytes(roster)
            tmpx = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
            tmpx.write(xlsx_bytes)
            tmpx.flush()
            tmpx.close()
            xlsx_token = uuid.uuid4().hex
            with DOWNLOAD_LOCK:
                DOWNLOAD_STORE[xlsx_token] = {"path": tmpx.name, "type": "xlsx", "filename": "roster_output.xlsx"}
            _cleanup_token_later(xlsx_token, delay_seconds=3600)
        except Exception:
            xlsx_token = None

    return jsonify({"status": "ok", "rows": rows, "csv_token": csv_token, "xlsx_token": xlsx_token})


@app.route("/download/csv")
def download_csv():
    token = request.args.get("token")
    if not token:
        flash("Missing token", "warning")
        return redirect(url_for("index"))
    with DOWNLOAD_LOCK:
        item = DOWNLOAD_STORE.get(token)
    if not item or not os.path.exists(item["path"]):
        flash("CSV not available", "warning")
        return redirect(url_for("index"))
    return send_file(item["path"], as_attachment=True, download_name=item.get("filename", "roster_output.csv"))

@app.route("/download/xlsx")
def download_xlsx():
    token = request.args.get("token")
    if not token:
        flash("Missing token", "warning")
        return redirect(url_for("index"))
    with DOWNLOAD_LOCK:
        item = DOWNLOAD_STORE.get(token)
    if not item or not os.path.exists(item["path"]):
        flash("XLSX not available", "warning")
        return redirect(url_for("index"))
    return send_file(item["path"], as_attachment=True, download_name=item.get("filename", "roster_output.xlsx"))


if __name__ == "__main__":
    app.run(debug=True)
