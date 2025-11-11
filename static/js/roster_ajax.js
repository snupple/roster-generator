// AJAX client for roster generator
document.addEventListener("DOMContentLoaded", function () {
  const runBtn = document.getElementById("run-btn");
  const form = document.getElementById("run-form");
  const progress = document.getElementById("progress");
  const progressText = document.getElementById("progress-text");
  const tableBody = document.querySelector("#result-table tbody");
  const downloadsDiv = document.getElementById("downloads");
  const alertsDiv = document.getElementById("alerts");

  function showAlert(msg, type = "info") {
    alertsDiv.innerHTML = `<div class="alert alert-${type}">${msg}</div>`;
    setTimeout(() => { alertsDiv.innerHTML = ""; }, 8000);
  }

  function clearResults() {
    tableBody.innerHTML = "";
    downloadsDiv.innerHTML = "";
  }

  function renderRows(rows) {
    tableBody.innerHTML = "";
    for (const r of rows) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${r.day || ""}</td><td>${r.slot || ""}</td><td>${r.id || ""}</td><td>${r.assigned || ""}</td>`;
      tableBody.appendChild(tr);
    }
  }

  function makeDownloadLink(token, label, endpoint) {
    const a = document.createElement("a");
    a.href = `${endpoint}?token=${encodeURIComponent(token)}`;
    a.className = "btn btn-success btn-sm me-2";
    a.textContent = label;
    a.target = "_blank";
    return a;
  }

  runBtn.addEventListener("click", async () => {
    clearResults();
    alertsDiv.innerHTML = "";

    const sampleChoice = document.getElementById("sample_choice").value;
    const useOpt = document.getElementById("use_optimizer").checked;
    const minRest = document.getElementById("min_rest").value;

    const fd = new FormData();
    fd.append("sample_choice", sampleChoice || "");
    fd.append("use_optimizer", useOpt ? "on" : "");
    fd.append("min_rest", minRest || "8");

    // If no sample_choice, attach uploaded files
    if (!sampleChoice) {
      const peopleFile = document.getElementById("people_file").files[0];
      const shiftsFile = document.getElementById("shifts_file").files[0];
      const prefillFile = document.getElementById("prefill_file").files[0];
      if (!peopleFile || !shiftsFile) {
        showAlert("Please provide people and shifts CSV files or select a sample.", "warning");
        return;
      }
      fd.append("people_file", peopleFile);
      fd.append("shifts_file", shiftsFile);
      if (prefillFile) fd.append("prefill_file", prefillFile);
    }

    try {
      progress.style.display = "inline-block";
      progressText.textContent = "Running...";
      runBtn.disabled = true;

      const resp = await fetch("/api/run", {
        method: "POST",
        body: fd
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ message: "Unknown error" }));
        showAlert("Failed: " + (err.message || resp.statusText), "danger");
        progress.style.display = "none";
        runBtn.disabled = false;
        return;
      }

      const data = await resp.json();
      if (data.status !== "ok") {
        showAlert("Error: " + (data.message || "unknown"), "danger");
        progress.style.display = "none";
        runBtn.disabled = false;
        return;
      }

      renderRows(data.rows);

      downloadsDiv.innerHTML = "";
      if (data.csv_token) {
        downloadsDiv.appendChild(makeDownloadLink(data.csv_token, "Download CSV", "/download/csv"));
      }
      if (data.xlsx_token) {
        downloadsDiv.appendChild(makeDownloadLink(data.xlsx_token, "Download XLSX", "/download/xlsx"));
      }

      showAlert("Assignment complete", "success");
    } catch (err) {
      showAlert("Request failed: " + err.message, "danger");
    } finally {
      progress.style.display = "none";
      runBtn.disabled = false;
    }
  });
});
