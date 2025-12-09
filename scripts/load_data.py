from pathlib import Path
from typing import List, Union, Iterable

def load_data(
    root_dir: Union[str, Path] = ".",
    extensions: Iterable[str] = (".txt",),
    recursive: bool = True,
    skip_empty: bool = True,
) -> List[Union["langchain.schema.Document", dict]]:
    
    root = Path(root_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"root_dir must be an existing directory. Got: {root}")

    # Try to import langchain Document type
    DocumentCls = None
    try:
        from langchain.schema import Document as DocumentCls  # type: ignore
    except Exception:
        DocumentCls = None

    files = []
    if recursive:
        for ext in extensions:
            files.extend(root.rglob(f"*{ext}"))
    else:
        for ext in extensions:
            files.extend(root.glob(f"*{ext}"))

    docs = []
    for file_path in sorted(files):
        # Skip directories (just in case) and hidden files
        if file_path.is_dir() or file_path.name.startswith("."):
            continue

        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            # If there's an unreadable file, skip but warn in metadata via placeholder
            text = ""

        if skip_empty and not text.strip():
            continue

        metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "dir": str(file_path.parent),
        }

        if DocumentCls is not None:
            doc_obj = DocumentCls(page_content=text, metadata=metadata)
        else:
            doc_obj = {"page_content": text, "metadata": metadata}

        docs.append(doc_obj)

    return docs
