#!/usr/bin/env bash
set -e

APPDIR=AppDir
rm -rf "$APPDIR"
python -m pip install --upgrade pip
pip install -r requirements.txt
pyinstaller ai_trainer.spec
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications"
cp dist/ai-trainer-tool "$APPDIR/usr/bin/"
cat > "$APPDIR/usr/share/applications/ai-trainer-tool.desktop" <<EOF
[Desktop Entry]
Name=AI Trainer Tool
Exec=ai-trainer-tool
Type=Application
Categories=Utility;Development;
EOF
if command -v appimagetool >/dev/null; then
  appimagetool "$APPDIR" ai-trainer-tool.AppImage
else
  echo "appimagetool not installed; skipped AppImage build"
fi
