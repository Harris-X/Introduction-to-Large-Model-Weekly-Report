## WSL2配置

> https://blog.csdn.net/ww_khun/article/details/129410363

1. 修改完Ubuntu软件源后，出现 

   W: GPG error: http://mirrors.aliyun.com/ubuntu trusty-security InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY 40976EAF437D05B5 NO_PUBKEY 3B4FE6ACC0B21F32`

```
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 40976EAF437D05B5
```

​	https://zhuanlan.zhihu.com/p/158824489

​	https://www.cnblogs.com/ellisonzhang/p/14077527.html

2. WSL2 jupyter LAB 运行报错 [W 2023-10-14 10:39:30.207 LabApp] Could not determine jupyterlab build status without nodejs

```
conda install -c conda-forge nodejs
```

3. Error code: Wsl/Service/0x80072746

```
wsl --update
```



