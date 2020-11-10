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
````

```bash
# Customize colors
export PS1="\[\033[95m\]\u@\h \[\033[32m\]\W\[\033[33m\] [\$(git symbolic-ref --short HEAD 2>/dev/null)]\[\033[00m\]\$ "
```

### Screen management

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



### Execution

````bash
# Pipelining
program1 | program2 | program3

# Scrollable output
program1 | more
# or less, because less is more
program1 | less

# Pass parameters
./script <param1> <param2>
````



### Files

````bash
# count number of words
wc -l <filename>
````



### Variables

````bash
# Set a variable
VARNAME=123

# Access a variable
$VARNAME

# Test varibale
echo $VARNAME
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

````
# Commmon ESC remap
:imap jk <Esc>

````

