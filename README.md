# rock-climbing-team-bee

## Team Members
- Glasson Antoine
- Houssemaine Pascal
- Guigui Marceau
- Chew Benjamin

---

## Guide d'entraînement RL (Reinforcement Learning)

L'entraînement par renforcement du robot grimpeur Microban repose sur **MjLab** (MuJoCo accéléré sur GPU avec NVIDIA Warp) et l'algorithme PPO de **RSL-RL**.

### 1. Prérequis & Installation

Depuis la racine du projet, rendez-vous dans le dossier de l'environnement :

```bash
cd src/mjlab_microban
uv sync
```

### 2. Tâches d'escalade disponibles

Pour lister l'ensemble des environnements enregistrés :

```bash
uv run list-envs
```

Trois étapes d'apprentissage progressif (curriculum) sont configurées :

| Tâche | Description |
|---|---|
| `Mjlab-Climb-Microban-Stage1` | Mur niveau 1 : apprentissage d'une transition simple (main droite vers cible fixe). |
| `Mjlab-Climb-Microban-Stage2` | Mur niveau 1 : transition d'un membre libre vers une nouvelle prise. |
| `Mjlab-Climb-Microban-Stage3` | Murs aléatoires N1, N2 et N3 : transitions multiples sur l'ensemble de la voie. |

### 3. Visualisation préliminaire (Play)

Avant de lancer l'entraînement, vous pouvez vérifier le robot suspendu aux prises de départ avec un agent factice :

```bash
# Robot immobile dans sa pose de départ
uv run play Mjlab-Climb-Microban-Stage1 --agent zero

# Robot avec actions aléatoires
uv run play Mjlab-Climb-Microban-Stage1 --agent random
```

### 4. Lancer l'entraînement (Train)

Pour démarrer l'entraînement sur GPU (ex. 1024 environnements en parallèle) :

```bash
uv run train Mjlab-Climb-Microban-Stage1 --env.scene.num-envs 1024
```

*Options courantes :*
* `--env.scene.num-envs <N>` : Nombre d'environnements parallèles (ex: 512, 1024, 2048 selon la VRAM GPU).
* `--agent.max-iterations <N>` : Nombre d'itérations d'entraînement (défaut : 10 000).
* Les logs et checkpoints sont enregistrés dans `logs/rsl_rl/microban_climb/[DATE_HEURE]/`.

> [!NOTE]
> L'avertissement MuJoCo `WARNING: Attaching a spec with an empty prefix...` est purement informatif lors de l'assemblage dynamique des prises sur le mur et n'affecte en rien l'entraînement.

### 5. Rejouer une politique entraînée

Une fois l'entraînement terminé ou en cours, visualisez les performances de votre politique :

```bash
uv run play Mjlab-Climb-Microban-Stage1 --checkpoint-file logs/rsl_rl/microban_climb/<DOSSIER_RUN>/model_<NUMERO>.pt
```

### 6. Murs de prises (Walls)

Des murs préconfigurés et générés aléatoirement sont disponibles dans `src/mjlab_microban/walls/` :

* **Visualiser un mur dans MuJoCo** :
  ```bash
  uv run climbing-sim-wall show walls/wall_n3_seed1.json
  ```
* **Éditer un mur dans le navigateur** (port 8765) :
  ```bash
  uv run climbing-sim-wall-editor walls/wall_n3_seed1.json
  ```
