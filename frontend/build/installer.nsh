; Custom NSIS script for richer installation detail output and progress text.

!include "LogicLib.nsh"

!ifndef BUILD_UNINSTALLER
  Var ArcRhoInstallProgressLastText

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
    StrCpy $ArcRhoInstallProgressLastText ""
    !insertmacro ArcRho_PrintInstallDetail "Installer progress monitoring started."
    !insertmacro ArcRho_PrintInstallDetail "Preparing destination and installing ArcRho files..."
    Call ArcRho_InstFiles_UpdateProgressText
    nsDialogs::CreateTimer ArcRho_InstFiles_UpdateProgressText 500
  FunctionEnd

  Function ArcRho_InstFiles_UpdateProgressText
    FindWindow $0 "#32770" "" $HWNDPARENT
    GetDlgItem $1 $0 1004
    GetDlgItem $2 $0 1006
    ${If} $1 == 0
    ${OrIf} $2 == 0
    ${OrIf} $0 == 0
      Return
    ${EndIf}

    SendMessage $1 0x0408 0 0 $3
    SendMessage $1 0x0407 0 0 $4
    SendMessage $1 0x0407 1 0 $5
    IntOp $8 $4 - $5
    ${If} $8 <= 0
      Return
    ${EndIf}

    IntOp $6 $3 - $5
    IntOp $6 $6 * 100
    IntOp $6 $6 / $8
    ${If} $6 < 10
      StrCpy $6 10
    ${ElseIf} $6 > 99
      StrCpy $6 99
    ${EndIf}

    StrCpy $7 "[$6%] Installing ArcRho files..."
    ${If} $7 != $ArcRhoInstallProgressLastText
      StrCpy $ArcRhoInstallProgressLastText $7
      SendMessage $2 0x000C 0 "STR:$7"
    ${EndIf}
  FunctionEnd

  Function ArcRho_InstFiles_CompleteProgressText
    FindWindow $0 "#32770" "" $HWNDPARENT
    GetDlgItem $1 $0 1006
    ${If} $0 == 0
    ${OrIf} $1 == 0
      Return
    ${EndIf}
    SendMessage $1 0x000C 0 "STR:[100%] Installation complete."
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
    nsDialogs::KillTimer ArcRho_InstFiles_UpdateProgressText
    Call ArcRho_InstFiles_CompleteProgressText
    !insertmacro ArcRho_PrintInstallDetail "[100%] Installation complete."
  !macroend
!endif

!macro customUnInstall
  SetDetailsPrint both
  SetDetailsView show
  DetailPrint "===== Uninstalling ArcRho ====="
!macroend
