"""Teacher scoring and coordinate-wise fusion."""

from .fusion import fuse_score_rows, normalize_weights
from .io import (
    FusedTrainingArtifact,
    TeacherScoreTable,
    TeacherTableError,
    fuse_teacher_tables_to_training_parquet,
    read_teacher_score_table,
)

__all__ = [
    "FusedTrainingArtifact",
    "TeacherScoreTable",
    "TeacherTableError",
    "fuse_score_rows",
    "fuse_teacher_tables_to_training_parquet",
    "normalize_weights",
    "read_teacher_score_table",
]
