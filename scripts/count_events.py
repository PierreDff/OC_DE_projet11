import requests

URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"

RECENT = "lastdate_end >= now(years=-1)"
DESCRIPTION = "description_fr is not null"

REQUETES = {
    "Lille commune":        'location_city = "Lille"',
    "Lille + 10 km":        "distance(location_coordinates, geom'POINT(3.0573 50.6292)', 10km)",
    "Lille + 15 km":        "distance(location_coordinates, geom'POINT(3.0573 50.6292)', 15km)",
}

for libelle, geo in REQUETES.items():
    for suffixe, clauses in [
        ("brut", [geo, RECENT]),
        ("+ description", [geo, RECENT, DESCRIPTION]),
    ]:
        where = " and ".join(clauses)
        r = requests.get(URL, params={"where": where, "limit": 1}, timeout=30)
        r.raise_for_status()
        print(f"{libelle:16} | {suffixe:14} | {r.json()['total_count']:>6}")
    print("-" * 48)