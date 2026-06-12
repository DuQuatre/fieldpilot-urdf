"""fieldpilot-urdf — import any ROS robot from a URL and run FK/IK/collision/
validation/repair/visualisation in pure Python. No ROS install, no build.

The open core of FieldPilot's robotics toolkit (AGPL-3.0). The gated SaaS adds
the LLM robot chat, the natural-language fault-diagnosis front-end, multi-tenant
hosting, and the spare-parts BOM — see https://github.com/DuQuatre/fieldpilot-urdf.

NOTE (scaffold): this package skeleton was created in PLAN.md task P1.2. The 13
open modules + `diagnose_core` are copied in from `pydexpi-server/app/urdf/` in
task P1.3b; until then the imports below are the *intended* public surface and
will resolve once the modules land. See PLAN.md Part 1.
"""

__version__ = "0.1.0.dev0"

# The intended OPEN public surface (populated by P1.3b). Mirrors the open subset
# of the SaaS package's __init__.py — the 12 pure-robotics modules plus the
# deterministic `diagnose_core` (diagnose / Symptom / Hypothesis / Verdict).
__all__ = [
    # loader
    "from_file", "from_xml", "to_xml",
    # models
    "Box", "Collision", "Cylinder", "Geometry", "Inertia", "Inertial",
    "Joint", "JointLimit", "Link", "Mesh", "Origin", "Robot", "Sphere", "Visual",
    # graph
    "build_graph", "chain", "is_tree", "joints_on_path", "leaf_links",
    "root_links", "subtree",
    # diagnostics
    "Finding", "RULES", "run_all", "summary",
    # faults (kinematic primitives — needed by diagnose_core)
    "inject_motor_fault", "freeze_joint", "fault_source_tag",
    # diagnose_core (the pure, deterministic two-tier loop)
    "diagnose", "Symptom", "Hypothesis", "Verdict", "DiagnoseReport",
    # fk
    "forward_kinematics", "joint_motion", "origin_to_T", "rotation_around_axis", "rpy_to_R",
    # collisions
    "aabb_overlap", "detect_self_collisions", "link_collision_aabbs", "transform_aabb",
    "MeshResolver", "clear_mesh_cache", "unresolved_meshes",
    # storage
    "save_robot", "load_robot", "load_meta", "load_resolver", "list_robots", "delete_robot",
    # repair
    "Patch", "REPAIRS", "UNFIXABLE_CODES", "repair",
    # trajectory
    "StepFinding", "WorkspaceResult", "check_trajectory", "sample_workspace", "trajectory_summary",
    # ik
    "IKResult", "solve_ik",
    # importer
    "expand_xacro", "fetch_meshes", "fetch_urdf", "import_urdf",
    "infer_package_root", "package_uri_parts", "resolve_includes", "substitute_find",
]
