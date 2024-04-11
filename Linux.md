
# 1
## 2
### 3
#### 4
##### 5
###### 6
*斜体文本*
_这也是斜体文本_

**粗体文本**
__这也是粗体文本__

`行内代码`
> quote

```
def print()
    block
```

###########################################  

## Linux
`find . -type f -name "*jpeg" -delete`

- specify Git info  
`git config --global user.email "you@example.com"`  
`git config --global user.name "Your Name"`<br><br>

- Github create repo
Local PC ---> Git Bash ---> cd Project  
`ssh: git@github.com:Johnny0217/crypto.git`
[ssh](https://github.com/Johnny0217/crypto.git)  
`$ git init`  
`$ git add .`<br> 
`$ git add *.py *.ipynb`<br>
`$ git add **/*.py **/*.ipynb`<br>
`$ git remote add origin https://github.com/Johnny0217/crypto.git`<br>
`$ git commit -m "message"`<br>
`$ git branch -M main`<br>
`$ git push -u origin master`<br><br>      
  
- when something update  
`$ git add .`  
`$ git commit -m "message"`  
`$ git push -u origin master`<br><br>  
  
- when you back to another device - fetch & merge = pull  
`$ git pull origin master`<br><br>
  
- Github in another device
`$ git clone ssh link`<br><br>