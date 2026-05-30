"""
knowledge_base.py — Maintenance Procedure Knowledge Base
=========================================================
Simulates what a real MRO (Maintenance, Repair, Overhaul) document database
would contain. In production this would be GKN's actual ATA-chapter documentation.

We use simple keyword-based retrieval (no vector DB needed) so it works fully offline.
The agent searches this KB to attach the correct procedure to each work order.
"""

from dataclasses import dataclass, field


# ── Maintenance Procedure Dataclass ──────────────────────────────────────

@dataclass
class MaintenanceProcedure:
    id: str
    title: str
    ata_chapter: str      # ATA 100 chapter (standard aerospace system coding)
    failure_modes: list[str]   # which failure modes trigger this procedure
    severity: str         # CRITICAL / HIGH / MEDIUM / LOW
    man_hours: float
    shop_visit: bool      # requires workshop (vs line maintenance)
    interval_cycles: int  # how often this should be done proactively
    steps: list[str]
    parts_to_preorder: list[str]


# ── Knowledge Base ────────────────────────────────────────────────────────

MAINTENANCE_PROCEDURES: list[MaintenanceProcedure] = [

    MaintenanceProcedure(
        id="MRO-001",
        title="High Pressure Compressor (HPC) Degradation — Borescope Inspection",
        ata_chapter="ATA 72-30",
        failure_modes=["HPC degradation", "compressor fouling", "blade erosion", "tip clearance loss"],
        severity="HIGH",
        man_hours=8.0,
        shop_visit=False,
        interval_cycles=500,
        steps=[
            "1. Cool engine to ambient temperature (min 4 hours after shutdown).",
            "2. Access HPC via borescope ports — stations 2.5, 3.5, 4.5.",
            "3. Inspect S1–S7 rotor blades for: erosion, FOD damage, cracking, tip curl.",
            "4. Measure tip clearance at each stage — serviceable limit 0.45 mm.",
            "5. Photograph any defects > 0.2 mm depth for engineering disposition.",
            "6. Check compressor wash records — schedule wash if > 200 cycles since last.",
            "7. Record findings in CAMP/aircraft logbook.",
        ],
        parts_to_preorder=["HPC blade set S3", "compressor wash kit", "borescope access plugs"],
    ),

    MaintenanceProcedure(
        id="MRO-002",
        title="High Pressure Turbine (HPT) Blade Inspection — Thermal Barrier Coating",
        ata_chapter="ATA 72-50",
        failure_modes=["HPT degradation", "turbine erosion", "thermal fatigue", "oxidation"],
        severity="CRITICAL",
        man_hours=24.0,
        shop_visit=True,
        interval_cycles=1500,
        steps=[
            "1. Remove HPT module — engine must be at MRO facility.",
            "2. Dimensional check of blade chord, width, and platform.",
            "3. Fluorescent penetrant inspection (FPI) on all HPT blades.",
            "4. TBC (Thermal Barrier Coating) thickness check — minimum 125 μm.",
            "5. Cooling hole flow check — all holes must flow within ±10% spec.",
            "6. Blade weight-moment sorting for re-assembly balance.",
            "7. Repair or replacement per engineering disposition document.",
        ],
        parts_to_preorder=["HPT blade set", "blade platform seals", "HPT shroud segments"],
    ),

    MaintenanceProcedure(
        id="MRO-003",
        title="Fan Blade Erosion and FOD (Foreign Object Damage) Assessment",
        ata_chapter="ATA 72-10",
        failure_modes=["fan degradation", "FOD impact", "leading edge erosion", "fan imbalance"],
        severity="HIGH",
        man_hours=6.0,
        shop_visit=False,
        interval_cycles=300,
        steps=[
            "1. Visual inspection of all fan blades — check for nicks, dents, leading-edge erosion.",
            "2. Measure any nicks — serviceable limit 1.5 mm depth on leading edge.",
            "3. Blend repair of minor damage per AMM 72-10-05.",
            "4. Post-repair balance check on fan trim balance equipment.",
            "5. If any blade exceeds serviceable limits → replace full blade set.",
        ],
        parts_to_preorder=["fan blade blending kit", "leading edge repair compound", "fan balance weights"],
    ),

    MaintenanceProcedure(
        id="MRO-004",
        title="Oil System Health — Bearing Wear and Debris Analysis",
        ata_chapter="ATA 72-80",
        failure_modes=["bearing wear", "oil contamination", "metal debris", "vibration"],
        severity="HIGH",
        man_hours=4.0,
        shop_visit=False,
        interval_cycles=200,
        steps=[
            "1. Drain oil and take sample for SOAP (Spectrometric Oil Analysis Program).",
            "2. Cut open oil filter — inspect for metallic particles.",
            "3. Check MFOD (Magnetic Ferrograph Oil Detector) readout.",
            "4. Chip detector inspection — if activated, engine is grounded.",
            "5. Borescope inspection of bearing compartment #1, #2, #3.",
            "6. Compare SOAP results against trend — any spike > 2 ppm increase = investigation.",
        ],
        parts_to_preorder=["oil filter assembly", "chip detector", "oil sample kit"],
    ),

    MaintenanceProcedure(
        id="MRO-005",
        title="Scheduled Engine Performance Restoration (PR) — Shop Visit",
        ata_chapter="ATA 72-00",
        failure_modes=["general degradation", "EGT margin loss", "thrust loss", "fuel burn increase"],
        severity="MEDIUM",
        man_hours=2000.0,
        shop_visit=True,
        interval_cycles=3000,
        steps=[
            "1. Full engine disassembly to module level.",
            "2. Each module inspected and repaired to new-serviceable limits.",
            "3. HPT and LPT blade replacement (if beyond repair).",
            "4. HPC restagger / trim — restore surge margin.",
            "5. Full reassembly, acceptance test cell run.",
            "6. EGT margin target: ≥ 50°C above redline at 30,000 ft ISA.",
        ],
        parts_to_preorder=["HPT vanes", "LPT blades", "combustion liner", "compressor seal set"],
    ),

    MaintenanceProcedure(
        id="MRO-006",
        title="Emergency Engine Shutdown and Quarantine Protocol",
        ata_chapter="ATA 72-00",
        failure_modes=["imminent failure", "critical degradation"],
        severity="CRITICAL",
        man_hours=1.0,
        shop_visit=True,
        interval_cycles=0,
        steps=[
            "1. GROUND AIRCRAFT IMMEDIATELY — do not operate until cleared.",
            "2. Notify maintenance control and engineering.",
            "3. Secure engine with intake and exhaust plugs.",
            "4. Preserve oil sample and oil filter for analysis.",
            "5. Download and preserve engine ACARS/health monitoring data.",
            "6. Initiate AOG (Aircraft on Ground) spare engine sourcing.",
        ],
        parts_to_preorder=["spare engine (lease)", "engine stand", "preservation kit"],
    ),
]


# ── Retrieval Function ────────────────────────────────────────────────────

def retrieve_procedures(
    failure_mode: str,
    rul_cycles: float,
    top_k: int = 2
) -> list[MaintenanceProcedure]:
    """
    Simple keyword-based retrieval.
    In production: replace with ChromaDB vector search over your ATA manuals.
    """
    failure_mode_lower = failure_mode.lower()

    # Emergency override — if RUL is critical, always include MRO-006
    results = []
    if rul_cycles < 15:
        emergency = next(p for p in MAINTENANCE_PROCEDURES if p.id == "MRO-006")
        results.append(emergency)

    # Score each procedure by keyword match
    scored = []
    for proc in MAINTENANCE_PROCEDURES:
        if proc.id == "MRO-006":
            continue
        score = 0
        for fm in proc.failure_modes:
            if any(word in failure_mode_lower for word in fm.lower().split()):
                score += 1
        if "hpc" in failure_mode_lower and "HPC" in proc.title:
            score += 2
        if "hpt" in failure_mode_lower and "HPT" in proc.title:
            score += 2
        if "fan" in failure_mode_lower and "Fan" in proc.title:
            score += 2
        scored.append((score, proc))

    scored.sort(key=lambda x: x[0], reverse=True)
    results += [p for _, p in scored[:top_k]]

    return results


def format_procedure_for_llm(proc: MaintenanceProcedure) -> str:
    """Format a procedure as context text for the LLM prompt."""
    return f"""
Procedure {proc.id}: {proc.title}
ATA Chapter: {proc.ata_chapter} | Severity: {proc.severity}
Man-hours: {proc.man_hours} | Shop visit required: {proc.shop_visit}
Recommended interval: every {proc.interval_cycles} cycles

Steps:
{chr(10).join(proc.steps)}

Parts to pre-order: {', '.join(proc.parts_to_preorder)}
""".strip()
