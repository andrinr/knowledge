# Ngnix

Ngnix is a open source webserver.



## Installation

[Read this guide][https://www.digitalocean.com/community/tutorials/how-to-install-nginx-on-ubuntu-18-04]

````bash
sudo apt update
sudo apt install nginx
sudo ufw app list
# should return something like that:
Available applications:
  Nginx Full
  Nginx HTTP
  Nginx HTTPS
  OpenSSH
````

If the output is inactive, try this:

````bash
sudo ufw enable
sudo ufw default deny
sudo iptables -L
````

Allow http and https connections:

````bash
sudo ufw allow 'Nginx Full'
````

## Checking status

````bash
systemctl status nginx
````

#### Reload configs:

```basg
nginx -s reload
```



[https://www.digitalocean.com/community/tutorials/how-to-install-nginx-on-ubuntu-18-04]: 

[https://www.digitalocean.com/community/tutorials/how-to-install-nginx-on-ubuntu-18-04]: 