# SSH

Create a new ssh key pair from powershell / bash on the client:

````powershell
ssh-keygen
````

Copy paste the public key from the client to the server in a authorized keys file:

````
scp key_id.pub user@remote.ch/.ssh/authorized_keys
````



## SHH for git

**Attention:** ssh only works with git when using the ssh url to clone repos, it will not work with the https url!

If you have generated a ssh key pari, simply add the public key to your github account.



