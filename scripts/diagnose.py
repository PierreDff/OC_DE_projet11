"""Élucidation des anomalies repérées par inspect_data.py."""

import json
from pathlib import Path

import pandas as pd

df = pd.DataFrame(json.loads(Path("data/raw/events_lille.json").read_text(encoding="utf-8")))

print("1. LIGNES SANS RÉGION — Lille en Belgique ?")
print("-" * 60)
sans_region = df[df["location_region"].isna()]
print(f"  {len(sans_region)} lignes")
print(sans_region[["title_fr", "location_name", "location_address"]].head(15).to_string())

print("\n\n2. TITRES LES PLUS RÉPÉTÉS")
print("-" * 60)
print(df["title_fr"].value_counts().head(10).to_string())

print("\n\n3. UN EXEMPLE DE TITRE RÉPÉTÉ, EN DÉTAIL")
print("-" * 60)
titre_frequent = df["title_fr"].value_counts().index[0]
extrait = df[df["title_fr"] == titre_frequent]
print(extrait[["uid", "daterange_fr", "location_name"]].head(8).to_string())
print("\n  Descriptions identiques ?",
      extrait["description_fr"].nunique() == 1)

print("\n\n4. ÉVÉNEMENTS TRÈS LOINTAINS (après 2028)")
print("-" * 60)
lointains = df[df["lastdate_end"] > "2028-01-01"]
print(f"  {len(lointains)} lignes")
print(lointains[["title_fr", "daterange_fr", "location_name"]].head(10).to_string())

print("\n\n5. LA LIGNE SANS TITRE / SANS DESCRIPTION")
print("-" * 60)
vides = df[df["title_fr"].isna() | df["description_fr"].isna()]
print(vides[["uid", "title_fr", "description_fr", "location_name"]].to_string())