from  socket import *
from threading import Thread
 
 
def recv_data():
    '''接受信息方法'''
    while True:
        print("---->")
        server_data = client_socket.recv(1024)
        server_connt = server_data.decode('gbk')
        print(f"对方说:{server_connt}")
        if server_connt == 'end':
            break
def send_data():
    '''发送信息方法'''
    while True:
        msg = input(">")
        client_socket.send(msg.encode('gbk'))
        if msg == 'end':
            print("结束聊天")
            break
 
if __name__ == '__main__':
    #创建套接字
    server_socket =socket(AF_INET,SOCK_STREAM)
    #连接端口
    server_socket.bind(('127.0.0.1',8887))  #ip可以不写
    #允许最大连接
    server_socket.listen(5)
    #接受
    client_socket,host = server_socket.accept()
    t1 = Thread(target=recv_data)
    t2 = Thread(target=send_data)
    t1.start()
    t2.start()
    #主线程等待t1、t2线程结束在结束
    t1.join()
    t2.join()
    
    client_socket.close()
    server_socket.close()
 
 
 
 