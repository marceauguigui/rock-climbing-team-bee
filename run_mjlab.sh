#!/bin/bash

# Se place automatiquement dans le bon dossier d'exécution
cd ~/rock-climbing-team-bee/src/mjlab_microban || { echo "Dossier introuvable"; exit 1; }

case "$1" in
    train)
        echo "🚀 Lancement d'un NOUVEL entraînement RL (à zéro)..."
        uv run train Mjlab-Climb-Microban-Stage1
        ;;
    resume)
        echo "♻️ Reprise de l'entraînement RL depuis le dernier checkpoint..."
        uv run train Mjlab-Climb-Microban-Stage1 --agent.resume True
        ;;
    play)
        # Trouve le checkpoint le plus récent proprement
        LATEST_PT=$(find logs/ -name "*.pt" -exec ls -t {} + 2>/dev/null | head -n 1)
        
        if [ -z "$LATEST_PT" ]; then
            echo "❌ Aucun fichier de sauvegarde (.pt) trouvé dans logs/"
            exit 1
        fi
        
        echo "🎮 Visualisation du modèle : $LATEST_PT"
        uv run play Mjlab-Climb-Microban-Stage1 --checkpoint-file="$LATEST_PT"
        ;;
    *)
        echo "Commandes disponibles :"
        echo "  ./run_mjlab.sh train   -> Lance un nouvel entraînement de zéro"
        echo "  ./run_mjlab.sh resume  -> Reprend l'entraînement existant (--agent.resume True)"
        echo "  ./run_mjlab.sh play    -> Joue automatiquement le checkpoint le plus récent"
        ;;
esac
