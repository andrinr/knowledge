# powershell

not even that bad

## Aliases

````powershell
Set-Alias -Name np -Value C:\Windows\notepad.exe

# With parameters

#1. define function
Function func-alias {ssh -i param a@b.ch}
Set-Alias -Name sshab -Value func-alias
````

