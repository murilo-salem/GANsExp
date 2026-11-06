"""Baixa recursivamente uma pasta do SharePoint (sesirs.sharepoint.com).

Autenticação: OAuth device-code interativo (suporta MFA via app de celular).
O terminal mostra um código; abra https://microsoft.com/devicelogin, digite o
código e autorize com a conta corporativa (aprovando a notificação MFA no
Microsoft Authenticator, se solicitado).

Uso:

    python3 login.py [--dest data/raw/safra_2025_2026] [--folder <caminho relativo>]
"""
import argparse
import json
import os
import sys
import time
import urllib.parse

import msal
import requests

SITE_URL = "https://sesirs.sharepoint.com/sites/DTI-CEDRAProgramaAgroSAT"
FOLDER_PATH = "Documentos Compartilhados/General/07_DOCUMENTOS_TECNICOS/Milho dados hist\u00f3ricos/Safra 2025-2026"

# Client ID do app nativo "Microsoft Office" (aceito para device flow com acesso SPO)
PUBLIC_CLIENT_ID = "d3590ed6-52b3-4102-aeff-aad2292ab01c"


def acquire_token(client_id):
    app = msal.PublicClientApplication(
        client_id=client_id,
        authority="https://login.microsoftonline.com/organizations",
    )
    accounts = app.get_accounts()
    if accounts:
        cached = app.acquire_token_silent(
            ["https://sesirs.sharepoint.com/.default"], account=accounts[0]
        )
        if cached and "access_token" in cached:
            print("Reutilizando token da sessão anterior.")
            return cached["access_token"]

    flow = app.initiate_device_flow(scopes=["https://sesirs.sharepoint.com/.default"])
    if "user_code" not in flow:
        raise RuntimeError(f"Falha ao iniciar device flow: {json.dumps(flow, indent=2)}")

    print("\n==============================================================")
    print("1) Abra https://microsoft.com/devicelogin no navegador")
    print("2) Digite o código abaixo e autorize com a conta SESI-RS:")
    print(f"\n        CÓDIGO: {flow['user_code']}\n")
    print("   Se o Microsoft Authenticator solicitar, aprove a notificação")
    print("   (ou digite o número de verificação exibido no app).")
    print("Este terminal fica aguardando...")
    print("==============================================================\n")
    sys.stdout.flush()

    result = app.acquire_token_by_device_flow(flow)
    if "access_token" not in result:
        raise RuntimeError(
            "Falha na autenticação: "
            f"{result.get('error')}: {result.get('error_description')}"
        )
    return result["access_token"]


def api_url(site_url, folder_path, endpoint):
    folder_enc = urllib.parse.quote(folder_path, safe="/")
    base = f"{site_url}/_api/web/GetFolderByServerRelativeUrl('{folder_enc}')"
    return f"{base}/{endpoint}"


def _retry(fn, retries=5, base_wait=3):
    last = None
    for attempt in range(retries):
        try:
            return fn()
        except requests.RequestException as e:
            last = e
            if e.response is not None and e.response.status_code in (429, 503):
                wait = base_wait * (attempt + 1)
                print(f"      (throttle {e.response.status_code}, aguardando {wait}s...)", flush=True)
                time.sleep(wait)
                continue
            raise
    raise last


def list_folder(token, site_url, folder_path):
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json;odata=nometadata",
    }
    folders = _retry(lambda: requests.get(api_url(site_url, folder_path, "Folders"), headers=headers, timeout=60))
    files = _retry(lambda: requests.get(api_url(site_url, folder_path, "Files"), headers=headers, timeout=60))
    subfolders = [f["ServerRelativeUrl"] for f in folders.json().get("value", [])]
    file_infos = [(f["Name"], f["Length"]) for f in files.json().get("value", [])]
    return subfolders, file_infos


def download_file(token, site_url, server_rel_url, dest):
    url = f"{site_url}/_api/web/GetFileByServerRelativeUrl('{urllib.parse.quote(server_rel_url, safe='/')}')/$value"
    headers = {"Authorization": f"Bearer {token}"}

    def _do():
        with requests.get(url, headers=headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    fh.write(chunk)
            return total

    return _retry(_do)


def download_recursive(token, site_url, folder_path, dest_root, indent=0):
    subfolders, file_infos = list_folder(token, site_url, folder_path)
    prefix = "  " * indent
    for name, size in file_infos:
        rel = os.path.join(dest_root, name)
        if os.path.exists(rel) and os.path.getsize(rel) == size:
            print(f"{prefix}[skip] {name} ({size} bytes)")
            continue
        print(f"{prefix}[get ] {name} ({size} bytes)", flush=True)
        try:
            download_file(token, site_url, f"{folder_path}/{name}", rel)
            print(f"{prefix}[done] {name}")
        except Exception as e:
            print(f"{prefix}[ERRO] {name}: {e}", file=sys.stderr)

    for sub in subfolders:
        rel_sub = sub.rstrip("/").split("/")[-1]
        print(f"{prefix}[dir ] {rel_sub}/")
        download_recursive(token, site_url, sub, os.path.join(dest_root, rel_sub), indent + 1)


def main():
    parser = argparse.ArgumentParser(description="Baixa pasta do SharePoint SESI-RS")
    parser.add_argument("--dest", default="data/raw/safra_2025_2026")
    parser.add_argument("--site", default=SITE_URL)
    parser.add_argument("--folder", default=FOLDER_PATH)
    parser.add_argument("--client-id", default=PUBLIC_CLIENT_ID)
    args = parser.parse_args()

    dest_root = os.path.abspath(args.dest)
    os.makedirs(dest_root, exist_ok=True)
    print(f"Site   : {args.site}")
    print(f"Pasta  : {args.folder}")
    print(f"Destino: {dest_root}")

    token = acquire_token(args.client_id)
    try:
        download_recursive(token, args.site, args.folder, dest_root)
    except Exception as e:
        print(f"\nERRO: {e}", file=sys.stderr)
        sys.exit(1)
    print("\nDownload concluído.")


if __name__ == "__main__":
    main()
