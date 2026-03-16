# PyInstaller spec for AI Trainer Tool
block_cipher = None

added_files = [
    ('src/gui/styles/light.qss', 'gui/styles'),
    ('src/gui/styles/dark.qss', 'gui/styles'),
    ('src/gui/styles/high_contrast.qss', 'gui/styles'),
]

a = Analysis(['src/main.py'],
             pathex=[],
             binaries=[],
             datas=added_files,
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='ai-trainer-tool',
          debug=False,
          strip=False,
          upx=True,
          console=False)
