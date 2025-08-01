from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import json
import threading
from typing import Dict, List
import uvicorn
from datetime import datetime

from poker_game import TreysPokerEnv, PokerAction
from dqn_agent import DQNAgent
from config import config
import numpy as np

app = FastAPI(title="DQN Poker API", version="1.0.0")

# CORS pour le frontend Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
training_active = False
connected_clients: List[WebSocket] = []

class GameManager:
    def __init__(self):
        try:
            print("🎮 Initialisation de l'environnement de jeu...")
            self.env = TreysPokerEnv()
            
            print("🤖 Initialisation de l'agent DQN...")
            self.agent = DQNAgent(
                state_size=self.env.state_size,
                num_actions=self.env.num_actions,
                learning_rate=config.DQN.LEARNING_RATE,
                epsilon=config.DQN.EPSILON_START,
                epsilon_min=config.DQN.EPSILON_MIN,
                epsilon_decay=config.DQN.EPSILON_DECAY,
                gamma=config.DQN.GAMMA,
                memory_size=config.DQN.MEMORY_SIZE
            )
            
            # Initialiser l'environnement avec un état de jeu
            print("🔄 Reset initial de l'environnement...")
            self.current_state = self.env.reset()
            
            self.training_stats = {
                "episode": 0,
                "total_reward": 0,
                "win_rate": 0,
                "epsilon": self.agent.epsilon,
                "memory_size": 0,
                "is_training": False
            }
            print("✅ GameManager initialisé avec succès")
            
        except Exception as e:
            print(f"❌ Erreur lors de l'initialisation du GameManager: {e}")
            raise e

# Initialisation sécurisée
try:
    game_manager = GameManager()
except Exception as e:
    print(f"❌ Erreur critique lors de l'initialisation: {e}")
    game_manager = None

# WebSocket pour les mises à jour en temps réel
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    
    try:
        while True:
            if game_manager:
                # Mettre à jour les stats actuelles
                game_manager.training_stats.update({
                    "epsilon": game_manager.agent.epsilon,
                    "memory_size": game_manager.agent.get_memory_size(),
                    "is_training": training_active
                })
                
                # Envoyer les stats actuelles
                await websocket.send_text(json.dumps({
                    "type": "stats_update",
                    "data": game_manager.training_stats
                }))
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)

async def broadcast_update(data: dict):
    """Diffuse une mise à jour à tous les clients connectés"""
    if connected_clients:
        message = json.dumps(data)
        disconnected_clients = []
        for client in connected_clients:
            try:
                await client.send_text(message)
            except:
                disconnected_clients.append(client)
        
        # Nettoyer les clients déconnectés
        for client in disconnected_clients:
            connected_clients.remove(client)

# API Endpoints

@app.get("/")
async def root():
    return {"message": "DQN Poker API is running", "game_manager_ready": game_manager is not None}

@app.get("/game/status")
async def get_game_status():
    """Retourne l'état actuel du jeu"""
    try:
        if not game_manager:
            return JSONResponse(content={"error": "Game manager not initialized"}, status_code=500)
        
        # Vérification de l'état de l'environnement
        if not hasattr(game_manager.env, 'player_chips'):
            # Réinitialiser si nécessaire
            game_manager.current_state = game_manager.env.reset()
        
        # Obtenir l'état actuel
        try:
            state = game_manager.env.get_state()
        except Exception as e:
            print(f"Erreur get_state: {e}")
            # Fallback: reset et retry
            game_manager.current_state = game_manager.env.reset()
            state = game_manager.env.get_state()
        
        return {
            "game_state": {
                "pot": int(getattr(game_manager.env, 'pot', 0)),
                "player_chips": int(getattr(game_manager.env, 'player_chips', 500)),
                "opponent_chips": int(getattr(game_manager.env, 'opponent_chips', 500)),
                "player_bet": int(getattr(game_manager.env, 'player_bet', 0)),
                "opponent_bet": int(getattr(game_manager.env, 'opponent_bet', 0)),
                "betting_round": getattr(game_manager.env, 'betting_round', 0),
                "community_cards": [str(card) for card in getattr(game_manager.env, 'community_cards', [])],
                "player_cards": [str(card) for card in getattr(game_manager.env, 'player_cards', [])],
                "done": getattr(game_manager.env, 'done', False),
                "winner": getattr(game_manager.env, 'winner', None)
            },
            "agent_stats": {
                "epsilon": float(game_manager.agent.epsilon),
                "memory_size": game_manager.agent.get_memory_size(),
                "q_values": game_manager.agent.get_q_values(state).tolist() if state is not None else []
            }
        }
    except Exception as e:
        print(f"❌ Erreur dans get_game_status: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": f"Game status error: {str(e)}"}, status_code=500)

@app.post("/game/reset")
async def reset_game():
    """Démarre une nouvelle partie"""
    try:
        if not game_manager:
            return JSONResponse(content={"error": "Game manager not initialized"}, status_code=500)
            
        state = game_manager.env.reset()
        game_manager.current_state = state
        return {"message": "Nouvelle partie démarrée", "initial_state": state.tolist() if state is not None else []}
    except Exception as e:
        print(f"❌ Erreur dans reset_game: {e}")
        return JSONResponse(content={"error": f"Reset error: {str(e)}"}, status_code=500)

@app.post("/game/action/{action_id}")
async def make_action(action_id: int):
    """Fait jouer l'agent avec l'action spécifiée"""
    try:
        if not game_manager:
            return JSONResponse(content={"error": "Game manager not initialized"}, status_code=500)
            
        if action_id not in range(5):
            return JSONResponse(content={"error": "Action invalide"}, status_code=400)
        
        # État avant l'action
        state = game_manager.env.get_state()
        
        # Exécution de l'action
        next_state, reward, done, info = game_manager.env.step(action_id)
        
        # Stockage de la transition
        if state is not None:
            game_manager.agent.store_transition(state, action_id, reward, next_state, done)
        
        # Mise à jour des stats
        game_manager.training_stats.update({
            "total_reward": game_manager.training_stats["total_reward"] + reward,
            "epsilon": game_manager.agent.epsilon,
            "memory_size": game_manager.agent.get_memory_size()
        })
        
        # Diffuser la mise à jour
        await broadcast_update({
            "type": "game_update",
            "data": {
                "action": action_id,
                "reward": float(reward),
                "done": done,
                "info": info
            }
        })
        
        return {
            "action": action_id,
            "reward": float(reward),
            "done": done,
            "info": info
        }
    except Exception as e:
        print(f"❌ Erreur dans make_action: {e}")
        return JSONResponse(content={"error": f"Action error: {str(e)}"}, status_code=500)

@app.post("/game/ai_action")
async def ai_action():
    """Laisse l'IA jouer automatiquement"""
    try:
        if not game_manager:
            return JSONResponse(content={"error": "Game manager not initialized"}, status_code=500)
            
        state = game_manager.env.get_state()
        if state is not None:
            action = game_manager.agent.act(state, training=True)
            return await make_action(action)
        else:
            return JSONResponse(content={"error": "Invalid game state"}, status_code=400)
    except Exception as e:
        print(f"❌ Erreur dans ai_action: {e}")
        return JSONResponse(content={"error": f"AI action error: {str(e)}"}, status_code=500)

@app.post("/training/start")
async def start_training():
    """Démarre l'entraînement automatique"""
    global training_active
    
    if training_active:
        return {"message": "Entraînement déjà en cours"}
    
    training_active = True
    if game_manager:
        game_manager.training_stats["is_training"] = True
    
    return {"message": "Entraînement démarré"}

@app.post("/training/stop")
async def stop_training():
    """Arrête l'entraînement"""
    global training_active
    training_active = False
    if game_manager:
        game_manager.training_stats["is_training"] = False
    return {"message": "Entraînement arrêté"}

@app.get("/agent/stats")
async def get_agent_stats():
    """Retourne les statistiques détaillées de l'agent"""
    try:
        if not game_manager:
            return JSONResponse(content={"error": "Game manager not initialized"}, status_code=500)
            
        stats = game_manager.agent.get_training_stats()
        
        return {
            "epsilon": float(game_manager.agent.epsilon),
            "memory_size": game_manager.agent.get_memory_size(),
            "recent_rewards": stats.get('rewards', [])[-50:],
            "recent_losses": stats.get('losses', [])[-50:],
            "episode_lengths": stats.get('episode_lengths', [])[-50:],
            "training_active": training_active
        }
    except Exception as e:
        print(f"❌ Erreur dans get_agent_stats: {e}")
        return JSONResponse(content={"error": f"Stats error: {str(e)}"}, status_code=500)

@app.post("/agent/save")
async def save_agent():
    """Sauvegarde le modèle de l'agent"""
    try:
        if not game_manager:
            return JSONResponse(content={"error": "Game manager not initialized"}, status_code=500)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"checkpoints/web_model_{timestamp}.weights.h5"
        game_manager.agent.save_model(save_path)
        return {"message": f"Modèle sauvegardé: {save_path}"}
    except Exception as e:
        print(f"❌ Erreur dans save_agent: {e}")
        return JSONResponse(content={"error": f"Save error: {str(e)}"}, status_code=500)

@app.post("/agent/load")
async def load_agent(model_path: str = "checkpoints/dqn_treys_poker_final.weights.h5"):
    """Charge un modèle pré-entraîné"""
    try:
        if not game_manager:
            return JSONResponse(content={"error": "Game manager not initialized"}, status_code=500)
            
        game_manager.agent.load_model(model_path)
        return {"message": f"Modèle chargé: {model_path}"}
    except Exception as e:
        print(f"❌ Erreur dans load_agent: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=400)

if __name__ == "__main__":
    print("🚀 Démarrage du serveur API DQN Poker...")
    print("📡 WebSocket: ws://localhost:8000/ws")
    print("🌐 API Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)