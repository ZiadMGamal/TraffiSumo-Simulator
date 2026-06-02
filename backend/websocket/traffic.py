from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from backend.services.simulation_service import SimulationService


def register_websocket_routes(app: FastAPI) -> None:
    @app.websocket("/ws/traffic")
    async def websocket_traffic(websocket: WebSocket):
        await websocket.accept()
        service = SimulationService.get_instance()
        try:
            await service.stream(websocket, interval=0.1)
        except WebSocketDisconnect:
            pass
        finally:
            service.shutdown()

    @app.websocket("/ws/training")
    async def websocket_training(websocket: WebSocket):
        await websocket.accept()
        from backend.services.training_service import TrainingService

        service = TrainingService.get_instance()
        try:
            while True:
                await websocket.send_json(service.get_status())
                import asyncio

                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            pass
