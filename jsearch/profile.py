"""A person's profile: what jobs are ranked against.

Built once from their resumes (fetch_jobs.py --build-profile), kept in <data dir>/profile.yaml and edited by
hand after that. It holds no secrets but is personal, so it lives in the git-ignored data folder.
"""
import hashlib
import logging
import zipfile
from pathlib import Path

import yaml
from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger("jsearch.profile")
RESUME_TYPES = {".pdf", ".docx"}
HEADER = ("# Your job-matching profile. Edit freely: jobs are ranked against it on every run.\n"
          "# skills: weight 3 = core strength, 2 = solid, 1 = familiar. aliases: other spellings to match.\n"
          "# resumes: your resume variants; each job suggests the one whose focus terms it mentions most.\n")


class ProfileError(ValueError):
    """profile.yaml (or the resumes it is built from) can't be used; the message says why and where."""


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Resume(Strict):
    label: str = Field(min_length=1)
    file: str = ""
    focus: list[str] = []


class Profile(Strict):
    name: str
    experience_years: float = Field(ge=0)
    summary: str = ""
    target_roles: list[str] = Field(min_length=1)
    skills: dict[str, int] = Field(min_length=1)
    aliases: dict[str, list[str]] = {}
    resumes: list[Resume] = []

    @field_validator("skills")
    @classmethod
    def _weights(cls, v: dict[str, int]) -> dict[str, int]:
        bad = {k: w for k, w in v.items() if w not in (1, 2, 3)}
        if bad:
            raise ValueError(f"weights must be 1, 2 or 3: {bad}")
        if any(not k.strip() for k in v):
            raise ValueError("a skill name is empty")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> "Profile":
        unknown = sorted(set(self.aliases) - set(self.skills))
        if unknown:
            raise ValueError(f"aliases for skills that are not in skills: {unknown}")
        labels = [r.label for r in self.resumes]
        if len(labels) != len(set(labels)):
            raise ValueError(f"resume labels must be unique: {labels}")
        return self


def load_profile(path: Path) -> Profile | None:
    """The profile at path, None when there is none yet. Raises ProfileError listing every problem."""
    if not path.exists():
        return None
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ProfileError(f"{path} is not valid YAML: {e}") from None
    try:
        return Profile.model_validate(raw)
    except ValidationError as e:
        problems = [f"  {'.'.join(map(str, err['loc'])) or '(top level)'}: {err['msg']}" for err in e.errors()]
        raise ProfileError(f"{path} is invalid:\n" + "\n".join(problems)) from None


def profile_hash(profile: Profile) -> str:
    """Changes whenever the profile does, so cached Claude scores for an older profile aren't reused."""
    return hashlib.sha256(profile.model_dump_json().encode("utf-8")).hexdigest()[:16]


def write_profile(profile: Profile, path: Path) -> Path:
    """Write as YAML; never overwrite an existing profile (writes <name>.new.yaml next to it instead)."""
    target = path if not path.exists() else path.with_name(f"{path.stem}.new{path.suffix}")
    target.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(profile.model_dump(), sort_keys=False, allow_unicode=True, width=110)
    target.write_text(HEADER + body, encoding="utf-8")
    return target


# --- resumes -----------------------------------------------------------------------------------------------

def resume_text(path: Path) -> str:
    """Plain text of a .pdf or .docx resume."""
    if path.suffix.lower() == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages).strip()
    doc = Document(str(path))
    parts = [p.text for p in doc.paragraphs]
    parts += [cell.text for table in doc.tables for row in table.rows for cell in row.cells]
    return "\n".join(t for t in parts if t.strip()).strip()


def read_resumes(folder: Path) -> dict[str, str]:
    """{file name: text} for every readable .pdf/.docx in folder. Unreadable files are skipped with a warning."""
    if not folder.is_dir():
        raise ProfileError(f"{folder} is not a folder")
    texts = {}
    for path in sorted(p for p in folder.iterdir() if p.suffix.lower() in RESUME_TYPES):
        try:
            text = resume_text(path)
        except (PdfReadError, PackageNotFoundError, zipfile.BadZipFile, OSError) as e:
            logger.warning(f"skipping {path.name}: can't read it ({type(e).__name__}: {e})")
            continue
        if not text:
            logger.warning(f"skipping {path.name}: no text found (a scanned image?)")
            continue
        texts[path.name] = text
    if not texts:
        raise ProfileError(f"no readable .pdf or .docx resumes in {folder}")
    return texts
