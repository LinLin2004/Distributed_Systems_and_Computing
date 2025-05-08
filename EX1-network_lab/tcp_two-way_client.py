from socket import *
from threading import Thread
 
 
def recv_data():
    """接受信息"""
    while True:
        rev_data = server_socket.recv(1024)
        rev_connt = rev_data.decode('gbk')
        print(f"对方说{rev_connt}")
        if rev_connt == 'end':
            print("结束聊天")
            break
def send_data():
    """发信息"""
    while True:
        msg = input(">")
        server_socket.send(msg.encode('gbk'))
        if msg == 'end':
            break
 
 
if __name__ == '__main__':
    #创建套接字
    server_socket =socket(AF_INET,SOCK_STREAM)
    #连接
    server_socket.connect(('127.0.0.1',8887))
 
    t1 =Thread(target=recv_data)
    t2 =Thread(target=send_data)
 
    t1.start()
    t2.start()
    t1.join()
    t2.join()
 
    # 结束
    server_socket.close()
 
 
 
 