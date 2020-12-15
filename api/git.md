# git



## Basics

````bash
git config --global user.name "John Doe"
git config --global user.email johndoe@example.com

git status

# add all changes to commit
git add -A

# commmit changes and add message
git commit -m "commit message"

# pull changes
git pull

# push changes
git pull
````



## Stashing

Temporary save changes to reuse later

````bash
git stash add 
git stash pop
````



## Branching

````bash
# new branch
git branch <branchname>

# delete branch
git branch -d <branchname>

# switch to branch
git checkout <branchname>
````



## Add Files to .gitignore and remove them from remote

````bash
# Add statement to .gitignore i.e.
.idea

# Check file out from master
git checkout master -- .gitignore

# Remove file from tree
git rm --cached -r .idea
````



## Change remote branch url

````bash
git remote set-url origin <url>
````