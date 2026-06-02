from fastapi import APIRouter

from backend.services.simulation_service import SimulationService

router = APIRouter(prefix="/api/simulation", tags=["simulation"])


@router.post("/initialize")
def initialize_simulation(use_gui: bool = False, force_mock: bool = False):
    service = SimulationService.get_instance()
    service.initialize(use_gui=use_gui, force_mock=force_mock)
    return {
        "status": "initialized",
        "mode": service.mode,
        "agents": len(service.env.agent_ids) if service.env else 0,
    }


@router.get("/step")
def simulation_step():
    service = SimulationService.get_instance()
    if not service.env:
        service.initialize()
    return service.step()


@router.post("/shutdown")
def shutdown_simulation():
    SimulationService.get_instance().shutdown()
    return {"status": "shutdown"}
