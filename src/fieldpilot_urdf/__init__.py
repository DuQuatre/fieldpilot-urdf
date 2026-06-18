"""fieldpilot-urdf — import any ROS robot from a URL and run FK/IK/collision/
validation/repair/visualisation in pure Python. No ROS install, no build.

The open core of FieldPilot's robotics toolkit (AGPL-3.0). The gated SaaS adds
the LLM robot chat, the natural-language fault-diagnosis front-end, multi-tenant
hosting, and the spare-parts BOM — see https://github.com/DuQuatre/fieldpilot-urdf.

Note: the visualisation renderers live in :mod:`fieldpilot_urdf.viz` and are NOT
imported here, so the core install stays light — `pip install fieldpilot-urdf[viz]`
and `from fieldpilot_urdf.viz import render_kinematic_tree, render_pose_3d`.
"""

__version__ = "1.24.0.dev0"

from .collisions import (
    MeshResolver, Obstacle, aabb_overlap, box_obstacle, clear_mesh_cache,
    detect_obstacle_collisions, detect_self_collisions, link_collision_aabbs,
    sphere_obstacle, transform_aabb, unresolved_meshes,
)
from .diagnostics import Finding, RULES, run_all, summary
from .mesh import SUPPORTED_FORMATS, read_mesh_bounds
from .faults import (
    fault_source_tag, freeze_joint, freeze_joint_at, inject_motor_fault, misconfigure_limit,
)
from .diagnose_core import (
    DiagnoseReport, Hypothesis, Symptom, Verdict, diagnose,
)
from .fk import forward_kinematics, joint_motion, origin_to_T, R_to_rpy, rotation_around_axis, rpy_to_R
from .graph import build_graph, chain, is_tree, joints_on_path, leaf_links, root_links, subtree
from .loader import from_file, from_xml, to_xml
from .models import (
    Box, Collision, Cylinder, FrameRef, Geometry, Inertia, Inertial, Joint,
    JointLimit, Link, LoopClosure, Mesh, Origin, Robot, Sphere, Visual,
)
from .repair import Patch, REPAIRS, UNFIXABLE_CODES, repair
from .storage import (
    delete_robot, list_robots, load_meta, load_resolver, load_robot, save_robot,
)
from .ik import IKResult, solve_ik, solve_ik_collision_free, solve_ik_multi
from .kinematics import (
    SingularityReport, geometric_jacobian, jacobian_joints,
    joint_velocity_to_twist, manipulability, singularity_report,
)
from .importer import (
    expand_xacro, expand_xacro_local, fetch_meshes, fetch_urdf, import_urdf,
    import_urdf_file, infer_package_root, package_uri_parts, resolve_includes,
    resolve_includes_local, substitute_find,
)
from .trajectory import (
    StepFinding, WorkspaceResult, check_trajectory, sample_workspace,
    trajectory_summary,
)
from .planning import (
    PlanResult, path_length, plan_path, shorten_path,
)
from .cartesian import (
    CartesianPlanResult, interpolate_pose, plan_cartesian_path, pose_error,
)
from .retime import TimedTrajectory, time_parameterize
from .fault_propagation import (
    RootCauseCandidate, affected_links, criticality, rank_root_causes,
)
from .kinematic_diagnosis import (
    CalibrationResult, JointFaultCandidate, PoseObservation,
    calibrate_joint_offsets, localize_joint_fault,
)
from .differential_diagnosis import (
    BeliefState, Candidate, Question, QuestionScore, candidates_from_scores,
    next_question, rank_questions, update_beliefs,
)
from .case_base import (
    DiagnosticCase, SolutionStat, delete_case, fault_priors, list_cases,
    load_case, load_cases, recommend_solution, save_case, solution_stats,
)
from .report import (
    DiagnosticReport, ReportImage, SparePart, attach_simulation_illustrations,
    build_simulation_illustrations, photo_requests, render_report_html,
)

__all__ = [
    "__version__",
    "from_file", "from_xml", "to_xml",
    "Box", "Collision", "Cylinder", "Geometry", "Inertia", "Inertial",
    "Joint", "JointLimit", "Link", "Mesh", "Origin", "Robot", "Sphere", "Visual",
    "FrameRef", "LoopClosure",
    "build_graph", "chain", "is_tree", "joints_on_path", "leaf_links",
    "root_links", "subtree",
    "Finding", "RULES", "run_all", "summary",
    "inject_motor_fault", "freeze_joint", "freeze_joint_at", "misconfigure_limit", "fault_source_tag",
    "diagnose", "Symptom", "Hypothesis", "Verdict", "DiagnoseReport",
    "forward_kinematics", "joint_motion", "origin_to_T", "R_to_rpy", "rotation_around_axis", "rpy_to_R",
    "aabb_overlap", "detect_self_collisions", "link_collision_aabbs", "transform_aabb",
    "MeshResolver", "clear_mesh_cache", "unresolved_meshes",
    "Obstacle", "box_obstacle", "sphere_obstacle", "detect_obstacle_collisions",
    "read_mesh_bounds", "SUPPORTED_FORMATS",
    "save_robot", "load_robot", "load_meta", "load_resolver", "list_robots", "delete_robot",
    "Patch", "REPAIRS", "UNFIXABLE_CODES", "repair",
    "StepFinding", "WorkspaceResult", "check_trajectory", "sample_workspace", "trajectory_summary",
    "PlanResult", "plan_path", "shorten_path", "path_length",
    "plan_cartesian_path", "CartesianPlanResult", "interpolate_pose", "pose_error",
    "time_parameterize", "TimedTrajectory",
    "IKResult", "solve_ik", "solve_ik_multi", "solve_ik_collision_free",
    "geometric_jacobian", "jacobian_joints", "joint_velocity_to_twist",
    "manipulability", "singularity_report", "SingularityReport",
    "expand_xacro", "expand_xacro_local", "fetch_meshes", "fetch_urdf", "import_urdf",
    "import_urdf_file", "infer_package_root", "package_uri_parts", "resolve_includes",
    "resolve_includes_local", "substitute_find",
    "RootCauseCandidate", "affected_links", "criticality", "rank_root_causes",
    "localize_joint_fault", "JointFaultCandidate",
    "calibrate_joint_offsets", "PoseObservation", "CalibrationResult",
    "Candidate", "Question", "QuestionScore", "BeliefState",
    "rank_questions", "next_question", "update_beliefs", "candidates_from_scores",
    "DiagnosticCase", "SolutionStat", "save_case", "load_case", "load_cases",
    "list_cases", "delete_case", "fault_priors", "solution_stats", "recommend_solution",
    "DiagnosticReport", "ReportImage", "SparePart", "photo_requests", "render_report_html",
    "build_simulation_illustrations", "attach_simulation_illustrations",
]
