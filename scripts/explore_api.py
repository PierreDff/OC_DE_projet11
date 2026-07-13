import requests

URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"

r = requests.get(URL, params={"limit": 1}, timeout=30)
r.raise_for_status()
data = r.json()

print("Total d'enregistrements :", data["total_count"])
print("-" * 70)

for cle, valeur in data["results"][0].items():
    print(f"{cle:30} | {str(valeur)[:55]}")