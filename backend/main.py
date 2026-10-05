import sys
#!/usr/bin/env python3
"""Local HTTP backend for the TorchTPU & JAX Interactive Learning Lab."""

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from urllib.parse import parse_qs, urlparse

LAB_ROOT = Path(__file__).resolve().parent.parent
MODULES_DIR = (LAB_ROOT / "modules").resolve()
RECIPES_DIR = (LAB_ROOT / "recipes").resolve()
FRONTEND_DIR = (LAB_ROOT / "frontend").resolve()
SCRIPTS_DIR = (LAB_ROOT / "scripts").resolve()
PROGRESS_FILE = (LAB_ROOT / ".progress.json").resolve()

PROJECT_ID = os.environ.get("TPU_LAB_PROJECT", "torchtpu-learning-lab")
HOST = os.environ.get("TPU_LAB_HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", os.environ.get("TPU_LAB_PORT", "8765")))

SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_.-]+$")
SAFE_ZONE_RE = re.compile(r"^[a-z]+-[a-z]+[0-9]-[a-z]$")
SAFE_ACCEL_RE = re.compile(r"^v[0-9]+[a-z]*-[0-9]+$")


def build_isolated_env() -> dict[str, str]:
  env = os.environ.copy()
  env["CLOUDSDK_CORE_PROJECT"] = PROJECT_ID
  env["PYTHONUNBUFFERED"] = "1"
  return env


def verify_subpath(base: Path, target: Path) -> Path:
  resolved_base = base.resolve()
  resolved_target = target.resolve()
  if not str(resolved_target).startswith(str(resolved_base) + os.sep):
    raise ValueError("Path escapes base directory")
  return resolved_target


def load_progress() -> dict:
  if PROGRESS_FILE.exists():
    try:
      return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    except Exception:
      pass
  return {"completed_units": [], "quiz_scores": {}, "tpu_config": {
      "zone": "us-central1-b",
      "node_id": "tpu-lab-vm",
      "accelerator_type": "v6e-1",
  }}


def save_progress(data: dict) -> None:
  PROGRESS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_curriculum() -> list[dict]:
  modules = []
  if not MODULES_DIR.exists():
    return modules
  for mod_dir in sorted(MODULES_DIR.iterdir()):
    if not mod_dir.is_dir():
      continue
    meta_file = mod_dir / "module.json"
    if not meta_file.exists():
      continue
    try:
      meta = json.loads(meta_file.read_text(encoding="utf-8"))
      meta["id"] = mod_dir.name
      modules.append(meta)
    except Exception:
      continue
  return modules


def get_unit_payload(module_id: str, unit_id: str) -> dict:
  if not SAFE_ID_RE.match(module_id) or not SAFE_ID_RE.match(unit_id):
    raise ValueError("Invalid module or unit identifier")

  mod_dir = verify_subpath(MODULES_DIR, MODULES_DIR / module_id)
  meta_file = verify_subpath(mod_dir, mod_dir / "module.json")
  meta = json.loads(meta_file.read_text(encoding="utf-8"))

  unit_meta = None
  for u in meta.get("units", []):
    if u.get("id") == unit_id:
      unit_meta = u
      break
  if not unit_meta:
    raise FileNotFoundError("Unit not found")

  md_path = verify_subpath(mod_dir, mod_dir / unit_meta["file"])
  markdown_text = md_path.read_text(encoding="utf-8")

  lab_code = ""
  lab_file = unit_meta.get("lab_file", "")
  if lab_file:
    lab_path = verify_subpath(mod_dir, mod_dir / lab_file)
    if lab_path.exists():
      lab_code = lab_path.read_text(encoding="utf-8")

  return {
      "module_id": module_id,
      "module_title": meta.get("title", module_id),
      "unit": unit_meta,
      "markdown": markdown_text,
      "starter_code": lab_code,
  }


def query_tpu_status(zone: str) -> dict:
  if not SAFE_ZONE_RE.match(zone):
    raise ValueError("Invalid GCP zone format")
  env = build_isolated_env()
  vms = []
  queued = []
  err_msg = ""

  try:
    vm_proc = subprocess.run(
        [
            "gcloud", "compute", "tpus", "tpu-vm", "list",
            f"--project={PROJECT_ID}",
            f"--zone={zone}",
            "--format=json",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if vm_proc.returncode == 0 and vm_proc.stdout.strip():
      raw_vms = json.loads(vm_proc.stdout)
      for v in raw_vms:
        name = v.get("name", "").split("/")[-1]
        endpoints = v.get("networkEndpoints", [])
        ip = endpoints[0].get("ipAddress", "") if endpoints else ""
        vms.append({
            "name": name,
            "state": v.get("state", "UNKNOWN"),
            "acceleratorType": v.get("acceleratorType", ""),
            "runtimeVersion": v.get("runtimeVersion", ""),
            "ipAddress": ip,
            "spot": v.get("schedulingConfig", {}).get("spot", False)
            or v.get("schedulingConfig", {}).get("preemptible", False),
        })
    elif vm_proc.stderr:
      err_msg = vm_proc.stderr.strip()
  except Exception as exc:
    err_msg = str(exc)

  try:
    qr_proc = subprocess.run(
        [
            "gcloud", "alpha", "compute", "tpus", "queued-resources", "list",
            f"--project={PROJECT_ID}",
            f"--zone={zone}",
            "--format=json",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if qr_proc.returncode == 0 and qr_proc.stdout.strip():
      raw_qrs = json.loads(qr_proc.stdout)
      for q in raw_qrs:
        name = q.get("name", "").split("/")[-1]
        state = q.get("state", {}).get("state", "UNKNOWN")
        queued.append({"name": name, "state": state})
  except Exception:
    pass

  return {
      "project_id": PROJECT_ID,
      "zone": zone,
      "vms": vms,
      "queued_resources": queued,
      "error": err_msg,
  }


def execute_python_code(
    code: str,
    target: str,
    zone: str,
    node_id: str,
    capture_hlo: bool = False,
) -> dict:
  start_ts = time.time()
  env = build_isolated_env()

  if target == "local":
    with tempfile.TemporaryDirectory(prefix="tpu_lab_local_") as tmpdir:
      script_path = Path(tmpdir) / "lab_run.py"
      script_path.write_text(code, encoding="utf-8")
      if capture_hlo:
        env["XLA_FLAGS"] = f"--xla_dump_to={tmpdir}/hlo_dump --xla_dump_hlo_as_text"
      try:
        proc = subprocess.run(
            [os.environ.get("TPU_LAB_LOCAL_PYTHON", "/usr/local/google/home/asrarfarooq/.pyenv/versions/axlearn-env/bin/python3") if Path("/usr/local/google/home/asrarfarooq/.pyenv/versions/axlearn-env/bin/python3").exists() else sys.executable, "-u", str(script_path)],
            cwd=tmpdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        hlo_text = ""
        hlo_dir = Path(tmpdir) / "hlo_dump"
        if hlo_dir.exists():
          for f in sorted(hlo_dir.glob("*after_optimizations*.txt"))[:2]:
            hlo_text += f"=== {f.name} ===\n" + f.read_text(encoding="utf-8", errors="replace")[:12000] + "\n"
        return {
            "target": "local",
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "hlo_dump": hlo_text,
            "duration_sec": round(time.time() - start_ts, 3),
        }
      except subprocess.TimeoutExpired:
        return {
            "target": "local",
            "returncode": 124,
            "stdout": "",
            "stderr": "Execution timed out after 120 seconds.",
            "hlo_dump": "",
            "duration_sec": round(time.time() - start_ts, 3),
        }

  if not SAFE_ZONE_RE.match(zone) or not SAFE_ID_RE.match(node_id):
    raise ValueError("Invalid zone or TPU node ID")

  with tempfile.TemporaryDirectory(prefix="tpu_lab_remote_") as tmpdir:
    local_script = Path(tmpdir) / "remote_lab_run.py"
    local_script.write_text(code, encoding="utf-8")

    scp_proc = subprocess.run(
        [
            "gcloud", "compute", "tpus", "tpu-vm", "scp",
            str(local_script),
            f"{node_id}:/tmp/remote_lab_run.py",
            f"--project={PROJECT_ID}",
            f"--zone={zone}",
            "--quiet",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    if scp_proc.returncode != 0:
      return {
          "target": f"tpu ({node_id} @ {zone})",
          "returncode": scp_proc.returncode,
          "stdout": scp_proc.stdout,
          "stderr": f"SCP to TPU VM failed:\n{scp_proc.stderr}",
          "hlo_dump": "",
          "duration_sec": round(time.time() - start_ts, 3),
      }

    remote_cmd = (
        "rm -rf /tmp/xla_hlo_dump && mkdir -p /tmp/xla_hlo_dump && "
        "if [ -f ~/tpu-env/bin/activate ]; then source ~/tpu-env/bin/activate; fi && "
    )
    if capture_hlo:
      remote_cmd += (
          'export XLA_FLAGS="--xla_dump_to=/tmp/xla_hlo_dump --xla_dump_hlo_as_text" && '
      )
    remote_cmd += "python3 -u /tmp/remote_lab_run.py"

    ssh_proc = subprocess.run(
        [
            "gcloud", "compute", "tpus", "tpu-vm", "ssh",
            node_id,
            f"--project={PROJECT_ID}",
            f"--zone={zone}",
            f"--command={remote_cmd}",
            "--quiet",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )

    hlo_text = ""
    if capture_hlo and ssh_proc.returncode == 0:
      hlo_proc = subprocess.run(
          [
              "gcloud", "compute", "tpus", "tpu-vm", "ssh",
              node_id,
              f"--project={PROJECT_ID}",
              f"--zone={zone}",
              "--command=head -n 160 /tmp/xla_hlo_dump/*after_optimizations*.txt 2>/dev/null || head -n 160 /tmp/xla_hlo_dump/*.txt 2>/dev/null || true",
              "--quiet",
          ],
          env=env,
          capture_output=True,
          text=True,
          timeout=30,
          check=False,
      )
      hlo_text = hlo_proc.stdout

    return {
        "target": f"tpu ({node_id} @ {zone})",
        "returncode": ssh_proc.returncode,
        "stdout": ssh_proc.stdout,
        "stderr": ssh_proc.stderr,
        "hlo_dump": hlo_text,
        "duration_sec": round(time.time() - start_ts, 3),
    }


def run_infra_script(action: str, zone: str, accelerator_type: str, node_id: str) -> dict:
  if not SAFE_ZONE_RE.match(zone) or not SAFE_ID_RE.match(node_id):
    raise ValueError("Invalid zone or node ID")
  if not SAFE_ACCEL_RE.match(accelerator_type):
    raise ValueError("Invalid accelerator type")

  env = build_isolated_env()
  region = "-".join(zone.split("-")[:2])

  if action == "bootstrap":
    cmd = ["bash", str(SCRIPTS_DIR / "bootstrap_project.sh"), region]
  elif action == "provision":
    cmd = ["bash", str(SCRIPTS_DIR / "provision_spot_tpu.sh"), "create", zone, accelerator_type, node_id]
  elif action == "hunt":
    cmd = ["bash", str(SCRIPTS_DIR / "provision_spot_tpu.sh"), "hunt", zone, accelerator_type, node_id]
  elif action == "delete":
    cmd = ["bash", str(SCRIPTS_DIR / "provision_spot_tpu.sh"), "delete", zone, accelerator_type, node_id]
  elif action == "install_runtime":
    cmd = ["bash", str(SCRIPTS_DIR / "install_tpu_runtime.sh"), zone, node_id]
  else:
    raise ValueError("Unsupported infrastructure action")

  proc = subprocess.run(
      cmd,
      cwd=str(LAB_ROOT),
      env=env,
      capture_output=True,
      text=True,
      timeout=600,
      check=False,
  )
  return {
      "action": action,
      "returncode": proc.returncode,
      "stdout": proc.stdout,
      "stderr": proc.stderr,
  }


class LabRequestHandler(BaseHTTPRequestHandler):

  def log_message(self, fmt: str, *args) -> None:
    return

  def send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
    body = json.dumps(payload).encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", "application/json; charset=utf-8")
    self.send_header("Content-Length", str(len(body)))
    self.send_header("X-Content-Type-Options", "nosniff")
    self.end_headers()
    self.wfile.write(body)

  def do_GET(self) -> None:
    parsed = urlparse(self.path)
    path = parsed.path
    qs = parse_qs(parsed.query)

    try:
      if path == "/api/curriculum":
        self.send_json({
            "project_id": PROJECT_ID,
            "modules": load_curriculum(),
            "progress": load_progress(),
        })
        return

      if path == "/api/recipes":
        recipes = []
        if RECIPES_DIR.exists():
          for rf in sorted(RECIPES_DIR.iterdir()):
            if rf.is_file() and rf.suffix in (".py", ".sh"):
              code_text = rf.read_text(encoding="utf-8", errors="replace")
              first_line = ""
              for line in code_text.splitlines():
                s = line.strip().strip('"\'# ').strip()
                if s and not s.startswith("!/"):
                  first_line = s
                  break
              recipes.append({
                  "id": rf.name,
                  "filename": f"recipes/{rf.name}",
                  "title": first_line or rf.name,
                  "lang": "bash" if rf.suffix == ".sh" else "python",
                  "runnable": rf.suffix == ".py",
                  "code": code_text,
              })
        self.send_json({"recipes": recipes})
        return

      if path == "/api/unit":
        module_id = qs.get("module", [""])[0]
        unit_id = qs.get("unit", [""])[0]
        self.send_json(get_unit_payload(module_id, unit_id))
        return

      if path == "/api/tpu/status":
        zone = qs.get("zone", ["us-central1-b"])[0]
        self.send_json(query_tpu_status(zone))
        return

      if path == "/favicon.ico":
        rel_path = "favicon.svg"
      else:
        rel_path = "index.html" if path in ("", "/") else path.lstrip("/")
      static_file = verify_subpath(FRONTEND_DIR, FRONTEND_DIR / rel_path)
      if not static_file.exists() or not static_file.is_file():
        self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        return

      content = static_file.read_bytes()
      mime_type, _ = mimetypes.guess_type(str(static_file))
      self.send_response(HTTPStatus.OK)
      self.send_header("Content-Type", mime_type or "application/octet-stream")
      self.send_header("Content-Length", str(len(content)))
      self.send_header("X-Content-Type-Options", "nosniff")
      self.end_headers()
      self.wfile.write(content)
    except Exception as exc:
      self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

  def do_POST(self) -> None:
    parsed = urlparse(self.path)
    path = parsed.path
    length = int(self.headers.get("Content-Length", "0"))
    raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"

    try:
      body = json.loads(raw_body)
      if path == "/api/run":
        code = str(body.get("code", ""))
        target = str(body.get("target", "local"))
        zone = str(body.get("zone", "us-central1-b"))
        node_id = str(body.get("node_id", "tpu-lab-vm"))
        capture_hlo = bool(body.get("capture_hlo", False))
        result = execute_python_code(code, target, zone, node_id, capture_hlo)
        self.send_json(result)
        return

      if path == "/api/progress":
        progress = load_progress()
        unit_key = body.get("completed_unit")
        if unit_key and isinstance(unit_key, str):
          if unit_key not in progress["completed_units"]:
            progress["completed_units"].append(unit_key)
        if "tpu_config" in body and isinstance(body["tpu_config"], dict):
          progress["tpu_config"].update(body["tpu_config"])
        save_progress(progress)
        self.send_json(progress)
        return

      if path == "/api/tpu/action":
        action = str(body.get("action", ""))
        zone = str(body.get("zone", "us-central1-b"))
        accel = str(body.get("accelerator_type", "v6e-1"))
        node_id = str(body.get("node_id", "tpu-lab-vm"))
        self.send_json(run_infra_script(action, zone, accel, node_id))
        return

      self.send_json({"error": "Unknown endpoint"}, HTTPStatus.NOT_FOUND)
    except Exception as exc:
      self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def main() -> None:
  os.chmod(SCRIPTS_DIR / "bootstrap_project.sh", 0o755)
  os.chmod(SCRIPTS_DIR / "provision_spot_tpu.sh", 0o755)
  os.chmod(SCRIPTS_DIR / "install_tpu_runtime.sh", 0o755)
  server = ThreadingHTTPServer((HOST, PORT), LabRequestHandler)
  print(f"TorchTPU & JAX Learning Lab listening on http://{HOST}:{PORT} (Project: {PROJECT_ID})")
  server.serve_forever()


if __name__ == "__main__":
  main()
