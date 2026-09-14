"""Baixa os ultimos 5 arquivos .tif (prefixo "Ortho") adicionados ontem na pasta
"Ortomosaicos e arquivos tif" do SharePoint SESI-RS (Safra 2023-2024).

Autenticacao: OAuth device-code interativo (mesmo fluxo do login.py).
O terminal mostra um codigo; abra https://microsoft.com/devicelogin, digite o
codigo e autorize com a conta corporativa (aprovando MFA se solicitado).
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.parse

import msal
import requests

SITE_URL = "https://sesirs.sharepoint.com/sites/DTI-CEDRAProgramaAgroSAT"
FOLDER_PATH = (
    "Documentos Compartilhados/General/07_DOCUMENTOS_TECNICOS/"
    "Milho dados hist\u00f3ricos/Safra 2023 - 2024/Ortomosaicos e arquivos tif"
)
PUBLIC_CLIENT_ID = "d3590ed6-52b3-4102-aeff-aad2292ab01c"
DEFAULT_PREFIX = "Ortho"
TOKEN_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".msal_token_cache.json")


def _build_app(client_id):
    app = msal.PublicClientApplication(
        client_id=client_id,
        authority="https://login.microsoftonline.com/organizations",
    )
    cache = msal.SerializableTokenCache()
    if os.path.exists(TOKEN_CACHE):
        try:
            with open(TOKEN_CACHE, "r", encoding="utf-8") as fh:
                cache.deserialize(fh.read())
        except Exception as e:
            print(f"(aviso: nao foi possivel ler o cache de token: {e})", file=sys.stderr)
    app.token_cache = cache
    return app, cache


def _save_cache(cache):
    try:
        with open(TOKEN_CACHE, "w", encoding="utf-8") as fh:
            fh.write(cache.serialize())
        os.chmod(TOKEN_CACHE, 0o600)
    except Exception as e:
        print(f"(aviso: nao foi possivel salvar o cache de token: {e})", file=sys.stderr)


def acquire_token(client_id):
    app, cache = _build_app(client_id)
    accounts = app.get_accounts()
    if accounts:
        cached = app.acquire_token_silent(
            ["https://sesirs.sharepoint.com/.default"], account=accounts[0]
        )
        if cached and "access_token" in cached:
            print("Reutilizando token da sessao anterior.")
            return cached["access_token"]

    flow = app.initiate_device_flow(scopes=["https://sesirs.sharepoint.com/.default"])
    if "user_code" not in flow:
        raise RuntimeError(f"Falha ao iniciar device flow: {json.dumps(flow, indent=2)}")

    print("\n==============================================================")
    print("1) Abra https://microsoft.com/devicelogin no navegador")
    print("2) Digite o codigo abaixo e autorize com a conta SESI-RS:")
    print(f"\n        CODIGO: {flow['user_code']}\n")
    print("   Se o Microsoft Authenticator solicitar, aprove a notificacao")
    print("   (ou digite o numero de verificacao exibido no app).")
    print("Este terminal fica aguardando...")
    print("==============================================================\n")
    sys.stdout.flush()

    result = app.acquire_token_by_device_flow(flow)
    if "access_token" not in result:
        raise RuntimeError(
            "Falha na autenticacao: "
            f"{result.get('error')}: {result.get('error_description')}"
        )
    _save_cache(cache)
    return result["access_token"]


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


def list_files(token, site_url, folder_path):
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json;odata=nometadata",
    }
    folder_enc = urllib.parse.quote(folder_path, safe="/")
    url = f"{site_url}/_api/web/GetFolderByServerRelativeUrl('{folder_enc}')/Files"
    resp = _retry(lambda: requests.get(url, headers=headers, timeout=60))
    data = resp.json().get("value", [])
    return [
        {
            "name": f["Name"],
            "size": f["Length"],
            "modified": f.get("TimeLastModified"),
            "created": f.get("TimeCreated"),
            "server_rel_url": f.get("ServerRelativeUrl"),
        }
        for f in data
    ]


def _parse_iso(ts):
    if not ts:
        return None
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def download_file(token, site_url, server_rel_url, dest):
    url = (
        f"{site_url}/_api/web/GetFileByServerRelativeUrl("
        f"'{urllib.parse.quote(server_rel_url, safe='/')}')/$value"
    )
    headers = {"Authorization": f"Bearer {token}"}

    def _do():
        with requests.get(url, headers=headers, stream=True, timeout=300) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            done = 0
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    fh.write(chunk)
                    done += len(chunk)
            return total or done

    return _retry(_do)


def main():
    parser = argparse.ArgumentParser(
        description="Baixa os ultimos 5 .tif (Ortho) adicionados ontem do SharePoint"
    )
    parser.add_argument("--dest", default="data/raw/safra_2023_2024/orthomosaics")
    parser.add_argument("--site", default=SITE_URL)
    parser.add_argument("--folder", default=FOLDER_PATH)
    parser.add_argument("--client-id", default=PUBLIC_CLIENT_ID)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Prefixo do nome do arquivo (sensivel a caixa)")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--no-date-filter",
        action="store_true",
        help="Ignora o filtro por 'ontem' e pega os --limit mais recentes",
    )
    args = parser.parse_args()

    dest_root = os.path.abspath(args.dest)
    os.makedirs(dest_root, exist_ok=True)
    print(f"Site   : {args.site}")
    print(f"Pasta  : {args.folder}")
    print(f"Destino: {dest_root}")

    today = dt.date.today()
    yesterday = today - dt.timedelta(days=1)
    print(f"Hoje   : {today.isoformat()} | Ontem: {yesterday.isoformat()}")

    token = acquire_token(args.client_id)
    print("\nListando arquivos da pasta...", flush=True)
    files = list_files(token, args.site, args.folder)

    candidates = [
        f
        for f in files
        if f["name"].startswith(args.prefix) and f["name"].lower().endswith(".tif")
    ]
    candidates.sort(
        key=lambda f: _parse_iso(f["modified"]) or dt.datetime.min.replace(tzinfo=dt.timezone.utc),
        reverse=True,
    )
    print(f"Total de arquivos na pasta: {len(files)}")
    print(f"Candidatos '{args.prefix}*.tif': {len(candidates)}")

    if not args.no_date_filter:
        yesterday_candidates = [
            f for f in candidates
            if (_parse_iso(f["modified"]) or _parse_iso(f["created"])).date() == yesterday
        ]
        print(f"Candidatos modificados/criados ontem ({yesterday.isoformat()}): {len(yesterday_candidates)}")
        selected = yesterday_candidates[: args.limit]
        if len(selected) < args.limit:
            print(
                f"Aviso: apenas {len(selected)} arquivo(s) correspondem a 'ontem'. "
                "Use --no-date-filter para baixar os mais recentes sem filtro de data."
            )
    else:
        selected = candidates[: args.limit]

    if not selected:
        print("\nNenhum arquivo para baixar. Saindo.")
        return

    print("\nArquivos selecionados para download:")
    for f in selected:
        mod = _parse_iso(f["modified"])
        mod_s = mod.astimezone().strftime("%Y-%m-%d %H:%M:%S") if mod else "?"
        print(f"  - {f['name']}  ({f['size']} bytes, mod: {mod_s})")

    print()
    for i, f in enumerate(selected, 1):
        dest = os.path.join(dest_root, f["name"])
        if os.path.exists(dest) and os.path.getsize(dest) == f["size"]:
            print(f"[{i}/{len(selected)}] [skip] {f['name']} (ja existe)")
            continue
        print(f"[{i}/{len(selected)}] [get ] {f['name']} ({f['size']} bytes)", flush=True)
        try:
            n = download_file(
                token,
                args.site,
                f["server_rel_url"] or f"{args.folder}/{f['name']}",
                dest,
            )
            print(f"[{i}/{len(selected)}] [done] {f['name']} ({n} bytes)")
        except Exception as e:
            print(f"[{i}/{len(selected)}] [ERRO] {f['name']}: {e}", file=sys.stderr)

    print("\nDownload concluido.")


if __name__ == "__main__":
    main()
