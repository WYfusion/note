注意常规情况下的Docker是已经被墙了，本文借鉴了[目前墙的情况下 在Ubuntu中安装docker最新的docker - 个人文章 - SegmentFault 思否](https://segmentfault.com/a/1190000044973692)和[docker被墙后无法pull的电友看过来 | 电鸭社区](https://eleduck.com/posts/Vvf4KB)在Ubuntu环境中配置Docker。

---
一、安装  
1、检查环境  
1.1 卸载旧版docker
```csharp
sudo su
apt remove docker docker-engine docker.io containerd runc
```
2、安装依赖
```sql
apt -y install ca-certificates curl gnupg lsb-release
```
3、添加密钥  
输入命令
```javascript
curl -fsSL http://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo apt-key add -
```
打印返回（返回OK即为成功）
```undefined
ok
```
4、添加Docker软件源
```bash
sudo add-apt-repository "deb [arch=amd64] http://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable"
```
```bash
sudo nano /etc/docker/daemon.json
```
将下面的镜像源添加到daemon.json中，若没有这个文件，则使用`sudo mkdir /etc/docker`先创建
```bash
{
  "registry-mirrors": [
    "https://hub.xdark.top",
    "https://hub.littlediary.cn",
    "https://dockerpull.org",
    "https://hub.crdz.gq",
    "http://docker.1panel.live",
    "https://docker.unsee.tech",
    "https://docker.udayun.com",
    "https://docker.kejilion.pro",
    "https://registry.dockermirror.com",
    "https://docker.rainbond.cc",
    "https://hub.geekery.cn",
    "https://docker.1panelproxy.com",
    "https://docker.linkedbus.com",
    "https://docker.nastool.de"
  ],
  "insecure-registries": ["harbor.test.com","registry.cn-shenzhen.aliyuncs.com"],
  "max-concurrent-downloads": 10
}
```

5、安装docker
```css
apt -y install docker-ce docker-ce-cli containerd.io
```
6、启动docker
```sql
systemctl start docker
```
7、查看docker状态
```bash
systemctl status docker
```
8、测试Docker
```bash
sudo docker run hello-world
```

9、重启docker服务
```bash
service docker restart
```