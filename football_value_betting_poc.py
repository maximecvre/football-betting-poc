"""
football_value_betting_poc.py
================================

Proof of Concept (PoC) : moteur de prédiction de résultats de football
basé sur du Machine Learning supervisé (XGBoost / Random Forest), couplé
à un module de détection de "value bets" et de dimensionnement de mise
via le Critère de Kelly.

Architecture du pipeline
-------------------------
1. simulate_raw_match_data()  -> génère (ou, en production, récupère via
   API-Football / OpenWeather / etc.) les données brutes d'un match.
2. build_features()           -> transforme les données brutes en
   variables exploitables par le modèle (feature engineering).
3. train_model()               -> entraîne un classifieur multi-classes
   (Domicile / Nul / Extérieur) et retourne des probabilités calibrées.
4. simulate_bookmaker_odds()   -> simule des cotes de bookmaker.
5. detect_value_bets()         -> croise probabilités IA et cotes, calcule
   l'espérance mathématique (EV) et la mise optimale (Critère de Kelly).

Ce script est un PoC : les données sont simulées avec numpy.random.
Pour brancher de vraies données, il suffit de remplacer le contenu de
`simulate_raw_match_data()` par des appels aux APIs réelles, en
conservant EXACTEMENT le même schéma de colonnes en sortie (voir les
docstrings de chaque fonction pour le contrat de données attendu).

Auteur : PoC généré pour usage éducatif / démonstration technique.
Avertissement : aucun modèle prédictif ne garantit un gain. Ce code est
fourni à des fins de démonstration technique (feature engineering,
pipeline ML, gestion de bankroll) et ne constitue pas un conseil
financier ou de pari.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# XGBoost est optionnel : on retombe sur RandomForest si absent, pour que
# le script reste exécutable "out of the box".
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGBOOST_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Table de correspondance météo -> impact estimé sur le nombre de buts.
# (valeur négative = tendance à réduire le nombre de buts marqués)
WEATHER_GOAL_IMPACT: dict[str, float] = {
    "sunny": 0.05,
    "cloudy": 0.0,
    "windy": -0.10,
    "rain": -0.20,
    "snow": -0.35,
}

RESULT_LABELS = ["AWAY_WIN", "DRAW", "HOME_WIN"]


# ---------------------------------------------------------------------------
# 1. SIMULATION / INGESTION DES DONNÉES BRUTES
# ---------------------------------------------------------------------------
def simulate_raw_match_data(n_matches: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Simule un jeu de données brut de matchs de football.

    En production, cette fonction serait remplacée par un connecteur vers
    des APIs réelles (API-Football pour les stats/blessures/arbitres,
    OpenWeather pour la météo, un fournisseur de cotes pour le bookmaker).
    Le schéma de colonnes retourné ci-dessous doit être préservé pour que
    `build_features()` reste compatible.

    Args:
        n_matches: nombre de matchs à simuler.
        seed: graine aléatoire pour la reproductibilité.

    Returns:
        DataFrame brut avec une ligne par match, incluant la variable
        cible `result` (HOME_WIN / DRAW / AWAY_WIN).
    """
    rng = np.random.default_rng(seed)

    # -- Variable latente non observable directement : écart de niveau
    # "réel" entre les deux équipes (sert uniquement à générer un signal
    # réaliste dans les données simulées, ce n'est PAS une feature fournie
    # au modèle).
    latent_strength_diff = rng.normal(0, 1.0, n_matches)

    # -- Repos & déplacements (pour calcul de la fatigue)
    rest_days_home = rng.integers(2, 12, n_matches)
    rest_days_away = rng.integers(2, 12, n_matches)
    travel_km_home = rng.uniform(0, 300, n_matches)       # domicile = peu de trajet
    travel_km_away = rng.uniform(50, 3000, n_matches)     # extérieur = trajet variable

    # -- Absences de joueurs clés : score d'impact composite (0 = aucun
    # impact, 1 = effectif décimé). Distribution asymétrique (bêta) car la
    # majorité des équipes ont peu de blessés importants.
    key_injury_impact_home = rng.beta(2, 8, n_matches)
    key_injury_impact_away = rng.beta(2, 8, n_matches)

    # -- Météo (catégorielle)
    weather_condition = rng.choice(
        list(WEATHER_GOAL_IMPACT.keys()),
        size=n_matches,
        p=[0.40, 0.25, 0.10, 0.20, 0.05],
    )

    # -- Historique de l'arbitre
    referee_red_card_avg = np.clip(rng.normal(2.0, 0.6, n_matches), 0.2, 5.0)
    referee_penalty_avg = np.clip(rng.normal(0.30, 0.12, n_matches), 0.0, 1.0)

    # -- Expected Goals (xG) : influencés par le niveau latent, la fatigue
    # et les blessures adverses, + bruit gaussien.
    fatigue_home = _fatigue_index(rest_days_home, travel_km_home)
    fatigue_away = _fatigue_index(rest_days_away, travel_km_away)

    xg_home = np.clip(
        1.35
        + 0.45 * latent_strength_diff
        - 0.35 * fatigue_home
        + 0.30 * key_injury_impact_away
        - 0.20 * key_injury_impact_home
        + rng.normal(0, 0.25, n_matches),
        0.05,
        None,
    )
    xg_away = np.clip(
        1.10
        - 0.45 * latent_strength_diff
        - 0.35 * fatigue_away
        + 0.30 * key_injury_impact_home
        - 0.20 * key_injury_impact_away
        + rng.normal(0, 0.25, n_matches),
        0.05,
        None,
    )

    df = pd.DataFrame(
        {
            "match_id": np.arange(n_matches),
            "rest_days_home": rest_days_home,
            "rest_days_away": rest_days_away,
            "travel_km_home": travel_km_home,
            "travel_km_away": travel_km_away,
            "key_injury_impact_home": key_injury_impact_home,
            "key_injury_impact_away": key_injury_impact_away,
            "weather_condition": weather_condition,
            "referee_red_card_avg": referee_red_card_avg,
            "referee_penalty_avg": referee_penalty_avg,
            "xg_home": xg_home,
            "xg_away": xg_away,
        }
    )

    # -- Génération de la variable cible à partir d'un score latent
    # (garde un lien statistique réaliste entre features et résultat,
    # indispensable pour qu'un modèle ait quelque chose à apprendre).
    df["result"] = _simulate_match_result(df, latent_strength_diff, rng)

    logger.info("Données brutes simulées : %d matchs générés.", n_matches)
    return df


def _fatigue_index(rest_days: np.ndarray, travel_km: np.ndarray) -> np.ndarray:
    """Calcule un indice de fatigue normalisé entre 0 (frais) et 1 (épuisé).

    Combine le manque de repos (moins de jours depuis le dernier match)
    et la distance parcourue (kilomètres de déplacement).
    """
    rest_component = 1 - np.clip((rest_days - 2) / 10, 0, 1)  # peu de repos -> proche de 1
    travel_component = np.clip(travel_km / 3000, 0, 1)
    return np.clip(0.6 * rest_component + 0.4 * travel_component, 0, 1)


def _simulate_match_result(
    df: pd.DataFrame, latent_strength_diff: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Simule le résultat réel du match à partir d'un score latent.

    Utilise une transformation softmax sur un score composite pour obtenir
    des probabilités par issue, puis tire le résultat au sort selon ces
    probabilités (approche multinomiale, plus réaliste qu'un simple seuil).
    """
    weather_impact = df["weather_condition"].map(WEATHER_GOAL_IMPACT).to_numpy()

    home_score = (
        0.9 * latent_strength_diff
        + 0.35 * (df["xg_home"] - df["xg_away"])
        + 0.5 * weather_impact  # la pluie/neige réduit l'avantage offensif global
    )
    draw_score = 0.15 - 0.25 * np.abs(latent_strength_diff)  # nul plus probable si équipes proches
    away_score = -home_score * 0.9  # anti-corrélé au score domicile

    logits = np.stack([away_score, draw_score, home_score], axis=1)
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    results = np.array(
        [rng.choice(RESULT_LABELS, p=p) for p in probs]
    )
    return results


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Construit les variables (features) prêtes pour l'entraînement du modèle.

    Cette fonction est le point d'entrée à adapter pour brancher de vraies
    sources de données : pour chaque bloc, si la colonne brute attendue est
    absente du DataFrame (= API non connectée), une valeur est simulée via
    numpy.random en fallback, afin que le pipeline reste exécutable de bout
    en bout même sans données réelles.

    Colonnes brutes attendues en entrée (si disponibles) :
        rest_days_home, rest_days_away, travel_km_home, travel_km_away,
        key_injury_impact_home, key_injury_impact_away, weather_condition,
        referee_red_card_avg, referee_penalty_avg, xg_home, xg_away

    Args:
        df: DataFrame brut (issu d'API réelles ou de `simulate_raw_match_data`).
        seed: graine utilisée uniquement pour les colonnes simulées en fallback.

    Returns:
        DataFrame de features numériques, une ligne par match, prêt pour
        `StandardScaler` puis l'entraînement du modèle.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    features = pd.DataFrame(index=df.index)

    # --- Indice de fatigue (repos + déplacement) ---
    if {"rest_days_home", "travel_km_home"}.issubset(df.columns):
        rest_days_home, travel_km_home = df["rest_days_home"], df["travel_km_home"]
    else:
        logger.warning("Données de fatigue (domicile) absentes -> simulation.")
        rest_days_home = rng.integers(2, 12, n)
        travel_km_home = rng.uniform(0, 300, n)

    if {"rest_days_away", "travel_km_away"}.issubset(df.columns):
        rest_days_away, travel_km_away = df["rest_days_away"], df["travel_km_away"]
    else:
        logger.warning("Données de fatigue (extérieur) absentes -> simulation.")
        rest_days_away = rng.integers(2, 12, n)
        travel_km_away = rng.uniform(50, 3000, n)

    features["fatigue_index_home"] = _fatigue_index(
        np.asarray(rest_days_home), np.asarray(travel_km_home)
    )
    features["fatigue_index_away"] = _fatigue_index(
        np.asarray(rest_days_away), np.asarray(travel_km_away)
    )
    features["fatigue_index_diff"] = (
        features["fatigue_index_away"] - features["fatigue_index_home"]
    )

    # --- Impact des absences de joueurs clés ---
    if "key_injury_impact_home" in df.columns:
        features["key_injury_impact_home"] = df["key_injury_impact_home"]
    else:
        logger.warning("Données de blessures (domicile) absentes -> simulation.")
        features["key_injury_impact_home"] = rng.beta(2, 8, n)

    if "key_injury_impact_away" in df.columns:
        features["key_injury_impact_away"] = df["key_injury_impact_away"]
    else:
        logger.warning("Données de blessures (extérieur) absentes -> simulation.")
        features["key_injury_impact_away"] = rng.beta(2, 8, n)

    features["injury_impact_diff"] = (
        features["key_injury_impact_away"] - features["key_injury_impact_home"]
    )

    # --- Météo : encodage numérique de l'impact sur le nombre de buts ---
    if "weather_condition" in df.columns:
        weather_condition = df["weather_condition"]
    else:
        logger.warning("Données météo absentes -> simulation.")
        weather_condition = pd.Series(
            rng.choice(list(WEATHER_GOAL_IMPACT.keys()), size=n)
        )
    features["weather_goal_impact"] = (
        weather_condition.map(WEATHER_GOAL_IMPACT).fillna(0.0).to_numpy()
    )

    # --- Historique de l'arbitre ---
    if "referee_red_card_avg" in df.columns:
        features["referee_red_card_avg"] = df["referee_red_card_avg"]
    else:
        logger.warning("Historique arbitre (cartons) absent -> simulation.")
        features["referee_red_card_avg"] = np.clip(rng.normal(2.0, 0.6, n), 0.2, 5.0)

    if "referee_penalty_avg" in df.columns:
        features["referee_penalty_avg"] = df["referee_penalty_avg"]
    else:
        logger.warning("Historique arbitre (penaltys) absent -> simulation.")
        features["referee_penalty_avg"] = np.clip(rng.normal(0.30, 0.12, n), 0.0, 1.0)

    # --- Statistiques avancées : Expected Goals (xG) ---
    if {"xg_home", "xg_away"}.issubset(df.columns):
        features["xg_home"] = df["xg_home"]
        features["xg_away"] = df["xg_away"]
    else:
        logger.warning("Données xG absentes -> simulation.")
        features["xg_home"] = np.clip(rng.normal(1.35, 0.4, n), 0.05, None)
        features["xg_away"] = np.clip(rng.normal(1.10, 0.4, n), 0.05, None)

    features["xg_diff"] = features["xg_home"] - features["xg_away"]

    return features


# ---------------------------------------------------------------------------
# 3. ENTRAÎNEMENT DU MODÈLE IA
# ---------------------------------------------------------------------------
@dataclass
class TrainedModel:
    """Conteneur pour un modèle entraîné et ses artefacts de préprocessing."""

    model: object
    scaler: StandardScaler
    label_encoder: LabelEncoder
    feature_names: list[str]
    test_accuracy: float = field(default=0.0)
    test_log_loss: float = field(default=0.0)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Retourne les probabilités par issue pour de nouvelles features.

        Args:
            X: DataFrame de features brutes (non standardisées), avec les
               mêmes colonnes que celles utilisées à l'entraînement.

        Returns:
            DataFrame avec une colonne par classe (AWAY_WIN, DRAW, HOME_WIN).
        """
        X_scaled = self.scaler.transform(X[self.feature_names])
        proba = self.model.predict_proba(X_scaled)
        # Réordonne les colonnes selon l'ordre du LabelEncoder pour être
        # certain de la correspondance classe <-> colonne.
        class_order = self.label_encoder.inverse_transform(
            np.arange(len(self.label_encoder.classes_))
        )
        return pd.DataFrame(proba, columns=class_order, index=X.index)


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    use_xgboost: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainedModel:
    """Entraîne un modèle de classification multi-classes (H/D/A).

    Pipeline : split train/test -> standardisation -> entraînement ->
    évaluation (accuracy + log loss) sur le jeu de test.

    Args:
        features: DataFrame de features numériques (sortie de `build_features`).
        target: Série des résultats réels ("HOME_WIN" / "DRAW" / "AWAY_WIN").
        use_xgboost: utilise XGBClassifier si disponible, sinon RandomForest.
        test_size: proportion du jeu de test.
        random_state: graine pour la reproductibilité du split et du modèle.

    Returns:
        Un `TrainedModel` prêt à produire des `predict_proba`.
    """
    feature_names = list(features.columns)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(target)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    # Normalisation : essentielle pour la stabilité numérique, même si les
    # modèles à base d'arbres (RF/XGBoost) y sont peu sensibles -- elle
    # facilite néanmoins le remplacement futur par un modèle linéaire/NN.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if use_xgboost and XGBOOST_AVAILABLE:
        logger.info("Entraînement d'un XGBClassifier.")
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=len(label_encoder.classes_),
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        if use_xgboost and not XGBOOST_AVAILABLE:
            logger.warning("xgboost non installé -> repli sur RandomForestClassifier.")
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_log_loss = log_loss(y_test, y_proba)

    logger.info(
        "Évaluation du modèle -> accuracy: %.3f | log loss: %.3f",
        test_accuracy,
        test_log_loss,
    )

    return TrainedModel(
        model=model,
        scaler=scaler,
        label_encoder=label_encoder,
        feature_names=feature_names,
        test_accuracy=test_accuracy,
        test_log_loss=test_log_loss,
    )


# ---------------------------------------------------------------------------
# 4. SIMULATION DES COTES BOOKMAKER
# ---------------------------------------------------------------------------
def simulate_bookmaker_odds(
    true_proba: pd.DataFrame, bookmaker_margin: float = 0.06, noise_scale: float = 0.04, seed: int = 11
) -> pd.DataFrame:
    """Simule des cotes de bookmaker (format décimal) à partir de probabilités.

    En production, cette fonction serait remplacée par un appel à une API
    de cotes (Odds API, Betfair Exchange, etc.). La simulation applique une
    marge bookmaker (overround) et un bruit pour représenter l'écart entre
    l'évaluation du bookmaker et celle de notre modèle -- c'est cet écart
    que le détecteur de value bets cherche à exploiter.

    Args:
        true_proba: probabilités "de référence" (ex: celles du modèle IA,
            utilisées ici uniquement pour générer des cotes réalistes).
        bookmaker_margin: marge globale du bookmaker (overround), typ. 5-8%.
        noise_scale: bruit appliqué pour simuler les divergences bookmaker/IA.
        seed: graine aléatoire.

    Returns:
        DataFrame de cotes décimales, mêmes colonnes que `true_proba`.
    """
    rng = np.random.default_rng(seed)
    noisy_proba = true_proba.to_numpy() + rng.normal(0, noise_scale, true_proba.shape)
    noisy_proba = np.clip(noisy_proba, 0.02, 0.98)
    noisy_proba = noisy_proba / noisy_proba.sum(axis=1, keepdims=True)

    # Overround : la somme des probabilités implicites des cotes dépasse 1.
    overround_proba = noisy_proba * (1 + bookmaker_margin)
    odds = 1 / overround_proba

    return pd.DataFrame(odds, columns=true_proba.columns, index=true_proba.index)


# ---------------------------------------------------------------------------
# 5. DÉTECTION DE VALUE BETS + CRITÈRE DE KELLY
# ---------------------------------------------------------------------------
def kelly_criterion(win_probability: float, decimal_odds: float, kelly_fraction: float = 0.5) -> float:
    """Calcule la fraction de bankroll à miser selon le Critère de Kelly.

    Formule : f* = (b*p - q) / b
        où b = cote nette (decimal_odds - 1)
           p = probabilité de gain estimée par le modèle
           q = 1 - p

    Args:
        win_probability: probabilité estimée (par l'IA) que le pari gagne.
        decimal_odds: cote décimale proposée par le bookmaker.
        kelly_fraction: fraction du Kelly "plein" à appliquer (0 < x <= 1).
            Une valeur < 1 (ex: 0.5 = "half Kelly") réduit la variance et
            le risque de ruine face à l'incertitude du modèle -- pratique
            standard en gestion de bankroll quantitative.

    Returns:
        Fraction de la bankroll à miser (0 si le Kelly plein est négatif).
    """
    b = decimal_odds - 1
    if b <= 0:
        return 0.0
    q = 1 - win_probability
    full_kelly = (b * win_probability - q) / b
    return max(0.0, full_kelly * kelly_fraction)


@dataclass
class ValueBet:
    """Représente une opportunité de pari à valeur positive détectée."""

    match_id: int
    outcome: str
    ai_probability: float
    bookmaker_odds: float
    expected_value: float
    kelly_stake_fraction: float
    recommended_stake: float


def detect_value_bets(
    ai_probabilities: pd.DataFrame,
    bookmaker_odds: pd.DataFrame,
    bankroll: float,
    ev_threshold: float = 0.0,
    kelly_fraction: float = 0.5,
    max_stake_fraction: float = 0.05,
) -> list[ValueBet]:
    """Croise probabilités IA et cotes bookmaker pour détecter les value bets.

    Pour chaque match et chaque issue (Domicile/Nul/Extérieur) :
        1. Calcule l'espérance mathématique :
               EV = (probabilité_IA * cote_décimale) - 1
        2. Si EV > ev_threshold, le pari est jugé "à valeur positive".
        3. La taille de la mise est déterminée par le Critère de Kelly
           (fractionné), plafonnée à `max_stake_fraction` de la bankroll
           pour limiter le risque en cas d'erreur du modèle.

    Args:
        ai_probabilities: probabilités prédites par le modèle IA
            (colonnes = issues, index = match_id).
        bookmaker_odds: cotes décimales du bookmaker (mêmes colonnes/index).
        bankroll: capital total disponible pour les mises.
        ev_threshold: seuil d'espérance minimal pour valider un pari.
        kelly_fraction: fraction du Kelly plein appliquée (gestion du risque).
        max_stake_fraction: plafond de sécurité (% max de la bankroll misé
            sur un seul pari), indépendant du résultat du calcul de Kelly.

    Returns:
        Liste des `ValueBet` détectés, triée par espérance décroissante.
    """
    value_bets: list[ValueBet] = []

    for match_id in ai_probabilities.index:
        for outcome in ai_probabilities.columns:
            p = float(ai_probabilities.loc[match_id, outcome])
            odds = float(bookmaker_odds.loc[match_id, outcome])

            expected_value = (p * odds) - 1

            if expected_value > ev_threshold:
                kelly_stake_fraction = kelly_criterion(p, odds, kelly_fraction)
                capped_fraction = min(kelly_stake_fraction, max_stake_fraction)
                recommended_stake = round(capped_fraction * bankroll, 2)

                if recommended_stake > 0:
                    value_bets.append(
                        ValueBet(
                            match_id=int(match_id),
                            outcome=outcome,
                            ai_probability=round(p, 4),
                            bookmaker_odds=round(odds, 3),
                            expected_value=round(expected_value, 4),
                            kelly_stake_fraction=round(capped_fraction, 4),
                            recommended_stake=recommended_stake,
                        )
                    )

    value_bets.sort(key=lambda vb: vb.expected_value, reverse=True)
    return value_bets


# ---------------------------------------------------------------------------
# 6. MOTEUR D'EXÉCUTION PRINCIPAL
# ---------------------------------------------------------------------------
def main(
    n_matches: int = 6000,
    bankroll: float = 1000.0,
    top_n_display: int = 10,
) -> None:
    """Exécute le pipeline complet de bout en bout, à titre de démonstration."""

    # 1. Données brutes (à remplacer par de vraies APIs en production)
    raw_df = simulate_raw_match_data(n_matches=n_matches)

    # 2. Feature engineering
    features = build_features(raw_df)
    target = raw_df["result"]

    # 3. Entraînement du modèle IA
    trained_model = train_model(features, target, use_xgboost=True)

    # 4. On simule un "nouveau" créneau de matchs à venir (jeu de test réutilisé
    #    ici pour l'exemple) sur lequel on applique le modèle + les cotes.
    upcoming_raw = simulate_raw_match_data(n_matches=200, seed=999)
    upcoming_features = build_features(upcoming_raw)

    ai_probabilities = trained_model.predict_proba(upcoming_features)
    ai_probabilities.index = upcoming_raw["match_id"]

    bookmaker_odds = simulate_bookmaker_odds(ai_probabilities)

    # 5. Détection des value bets + dimensionnement Kelly
    value_bets = detect_value_bets(
        ai_probabilities,
        bookmaker_odds,
        bankroll=bankroll,
        ev_threshold=0.0,
        kelly_fraction=0.5,       # half-Kelly : réduit la variance
        max_stake_fraction=0.05,  # jamais plus de 5% de la bankroll sur un pari
    )

    # --- Affichage synthétique ---
    print("\n" + "=" * 70)
    print(f"Modèle utilisé      : {'XGBoost' if XGBOOST_AVAILABLE else 'RandomForest'}")
    print(f"Accuracy (test)      : {trained_model.test_accuracy:.3f}")
    print(f"Log loss (test)      : {trained_model.test_log_loss:.3f}")
    print(f"Value bets détectés  : {len(value_bets)} / {len(upcoming_raw) * 3} issues évaluées")
    print("=" * 70)

    for vb in value_bets[:top_n_display]:
        print(
            f"Match #{vb.match_id:>4} | {vb.outcome:<9} | "
            f"P(IA)={vb.ai_probability:.3f} | cote={vb.bookmaker_odds:.2f} | "
            f"EV={vb.expected_value:+.3f} | mise={vb.recommended_stake:.2f}€ "
            f"({vb.kelly_stake_fraction:.1%} bankroll)"
        )

    if not value_bets:
        print("Aucune value bet détectée sur ce lot de matchs simulés.")

    print("=" * 70 + "\n")

    # Export CSV des value bets détectés (utile pour un run automatisé,
    # par exemple récupéré comme artifact GitHub Actions).
    if value_bets:
        vb_df = pd.DataFrame([vb.__dict__ for vb in value_bets])
    else:
        vb_df = pd.DataFrame(
            columns=[
                "match_id", "outcome", "ai_probability", "bookmaker_odds",
                "expected_value", "kelly_stake_fraction", "recommended_stake",
            ]
        )
    vb_df.to_csv("value_bets_output.csv", index=False)
    logger.info("Résultats exportés dans value_bets_output.csv")


if __name__ == "__main__":
    # Permet de surcharger les paramètres par défaut via variables
    # d'environnement (pratique pour un déclenchement depuis une CI/CD
    # comme GitHub Actions, sans avoir à modifier le code).
    import os

    n_matches_arg = int(os.environ.get("N_MATCHES", 6000))
    bankroll_arg = float(os.environ.get("BANKROLL", 1000.0))

    main(n_matches=n_matches_arg, bankroll=bankroll_arg)
