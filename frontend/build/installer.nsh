; Custom NSIS script for richer installation detail output and progress text.

!include "LogicLib.nsh"

!ifndef BUILD_UNINSTALLER
  !macro ArcRho_PrintInstallDetail MSG
    SetDetailsPrint both
    SetDetailsView show
    DetailPrint "${MSG}"
  !macroend
!endif

; Keep details visible by default so users can inspect installer actions.
ShowInstDetails show
ShowUninstDetails show

!macro preInit
  SetDetailsPrint both
!macroend

!macro customInit
  SetDetailsPrint both
  SetDetailsView show
  DetailPrint "===== Installing ArcRho ====="
  DetailPrint "Preparing installation..."
!macroend

; This hook is inserted immediately before MUI_PAGE_INSTFILES.
!ifndef BUILD_UNINSTALLER
  !macro customPageAfterChangeDir
    !define MUI_PAGE_CUSTOMFUNCTION_SHOW ArcRho_InstFiles_Show
  !macroend

  Function ArcRho_InstFiles_Show
    !insertmacro ArcRho_PrintInstallDetail "Installer progress monitoring started."
    !insertmacro ArcRho_PrintInstallDetail "[10%] Checking previous installation and preparing destination..."
  FunctionEnd

  ; electron-builder runs this after the embedded package is extracted.
  !macro customFiles_x64
    !insertmacro ArcRho_PrintInstallDetail "[65%] Core application files extracted."
    !insertmacro ArcRho_PrintInstallDetail "[75%] Writing installer metadata..."
  !macroend

  !macro customFiles_ia32
    !insertmacro ArcRho_PrintInstallDetail "[65%] Core application files extracted."
    !insertmacro ArcRho_PrintInstallDetail "[75%] Writing installer metadata..."
  !macroend

  !macro customFiles_arm64
    !insertmacro ArcRho_PrintInstallDetail "[65%] Core application files extracted."
    !insertmacro ArcRho_PrintInstallDetail "[75%] Writing installer metadata..."
  !macroend

  ; Optional no-op hook used only to surface a progress line between file extraction and final completion.
  !macro registerFileAssociations
    !insertmacro ArcRho_PrintInstallDetail "[90%] Creating shortcuts and registry entries..."
  !macroend

  !macro customInstall
    !insertmacro ArcRho_PrintInstallDetail "[100%] Installation complete."
  !macroend
!endif

!macro customUnInstall
  SetDetailsPrint both
  SetDetailsView show
  DetailPrint "===== Uninstalling ArcRho ====="
!macroend
