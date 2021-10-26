# bash

## General

### Navigation

````bash
# step into folder
cd folderName

# step out of folder
cd ../

# go to home
cd $HOME 
# or
cd ~

# list files
ls
# include hiden files
ls -A
# list files with certain ending <end>
ls *.<end>

# current linux version
hostnamectl

# check file site
du -hs <directory>
````

```bash
# Customize colors
export PS1="\[\033[95m\]\u@\h \[\033[32m\]\W\[\033[33m\] [\$(git symbolic-ref --short HEAD 2>/dev/null)]\[\033[00m\]\$ "
```

#### Xclip

````bash
sudo apt-get install xclip
# usage
cat file | xclip
````

### Screen management

#### screen

````bash
# start screen session:
screen
# split horizontal
ctrl + a, S
# split vertical 
ctrl + a, |
# switch split
ctrl + a, tab
# remove current split
ctrl + a, X
# new prompt 
ctrl + a, c
````

#### tmux

[cheat sheet](https://tmuxcheatsheet.com/)

considered superior to screen

Sessions:

````{bash}
# start tmux
tmux
# detach session
Ctrl + b, d
# come back to session
tmux attach
tmux a
# change to next window
c-b n
# change to previous window 
C-b p
# list all sessions
tmux ls
# kill all session but cuurrent
tmux kill-session -a
# kill specific session
tmux kill-session -t <number>
````

Panes

```{bash}
# vertical split
C-b %
# horizontal split 
C-b "
# Navigate 
C-b left | right | up | down
# close pane
C-d
# make pane fullscreen
C-b z

```



### Execution

````bash
# Pipelining
program1 | program2 | program3

# Scrollable output
program1 | more
# or less, because less is more
program1 | less
````

## Permissions

```{bash}
# User can read write and execute
# Group can read and execute
# Others can only read it
chmod u=rwx,g=rx,o=r myfile
```





### Files

````bash
# count number of words
wc -l <filename>
````



### Variables & Arguments

````bash
# Set a variable
VARNAME=123

# Access a variable
$VARNAME

# Test varibale
echo $VARNAME

# Pass parameters
./script <param1> <param2>

# Access parameters 1, 2
echo $1
echo $2
````



## VIM

### Navigation

````bash
# insert mode
i
# visual mode
v
# normal mode 
esc
# navigate
h j k l
# skip to next word, start of word
w
# skip to next word, end of word
e
# back to previous word
b
# first line
gg
# last line
G
# specific line
<number>G
# first character
0
````

### Editing

````bash
# delete characters
x
# delete characters
<amount>x
# undo
u
# undo n-times
nu
# delete word
dw
# delete liney copies it, can be pasted
dd
# copy line
y
# paste line
p
# delete word, enter insert mode
ce 
# alt + nav, navigate and leave edit mode

````

### Remapping

````bash
# Commmon ESC remap
:imap jk <Esc>

````

## Remote display

First launch XLaunch then open a WSL terminal or do it via Putty and enable X11 tunneling

````bash
# In WSL
DISPLAY=localhost:0
ssh -Y remote
````

