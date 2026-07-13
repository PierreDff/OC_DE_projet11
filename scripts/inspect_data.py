"""Inspection du jeu brut avant nettoyage : volumétrie, complétude, doublons."""

import json
from pathlib import Path

import pandas as pd

SOURCE = Path("data/raw/events_lille.json")

df = pd.DataFrame(json.loads(SOURCE.read_text(encoding="utf-8")))

print(f"Lignes : {len(df)}  |  Colonnes : {len(df.columns)}")
print("=" * 60)

print("\nTAUX DE REMPLISSAGE")
remplissage = df.notna().mean().sort_values() * 100
for colonne, taux in remplissage.items():
    print(f"  {colonne:22} {taux:5.1f} %")

print("\nDOUBLONS")
print(f"  uid dupliqués      : {df['uid'].duplicated().sum()}")
print(f"  titres dupliqués   : {df['title_fr'].duplicated().sum()}")

print("\nLONGUEUR DES TEXTES (en caractères)")
for colonne in ["title_fr", "description_fr", "longdescription_fr"]:
    longueurs = df[colonne].fillna("").astype(str).str.len()
    vides = (longueurs == 0).sum()
    print(
        f"  {colonne:22} médiane={longueurs.median():6.0f}  "
        f"max={longueurs.max():6.0f}  vides={vides}"
    )

print("\nDATES")
print(f"  lastdate_end min : {df['lastdate_end'].min()}")
print(f"  lastdate_end max : {df['lastdate_end'].max()}")

print("\nEXEMPLE DE longdescription_fr (500 premiers caractères)")
print("-" * 60)
exemple = df["longdescription_fr"].dropna().iloc[0]
print(exemple[:500])