import socket
import threading
import sys

# 配置参数
SERVER_ADDR = ('127.0.0.1', 1025)  # 服务器地址
BUFFER_SIZE = 1024                  # 缓冲区大小

def tcp_communication():
    """TCP单向通信主函数"""
    role = input("请选择角色(1-服务器/2-客户端): ")
    
    if role == '1':
        run_server()
    elif role == '2':
        run_client()
    else:
        print("无效选择")

def run_server():
    """TCP服务器端实现"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(SERVER_ADDR)
        server_socket.listen(1)
        print(f"[服务器] 正在 {SERVER_ADDR} 监听...")

        try:
            while True:
                conn, addr = server_socket.accept()
                print(f"[服务器] 接受来自 {addr} 的连接")
                
                with conn:
                    while True:
                        try:
                            # 接收数据
                            data = conn.recv(BUFFER_SIZE)
                            if not data:
                                break
                                
                            msg = data.decode('gbk')
                            print(f"\n[来自 {addr} 的消息]: {msg}")
                            
                            if msg.upper() == "BYE":
                                print("[服务器] 收到退出指令，关闭连接...")
                                break
                                
                        except ConnectionResetError:
                            print("[服务器] 客户端断开连接")
                            break
                            
        except KeyboardInterrupt:
            print("\n[服务器] 正在关闭...")

def run_client():
    """TCP客户端实现"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        try:
            client_socket.connect(SERVER_ADDR)
            print(f"[客户端] 已连接到服务器 {SERVER_ADDR}")
            
            try:
                while True:
                    # 发送数据
                    msg = input("[请输入消息]: ")
                    client_socket.sendall(msg.encode('gbk'))
                    
                    if msg.upper() == "BYE":
                        print("[客户端] 发送退出指令，关闭连接...")
                        break
                        
            except ConnectionResetError:
                print("[客户端] 服务器断开连接")
                
        except ConnectionRefusedError:
            print("[客户端] 无法连接到服务器")
        except KeyboardInterrupt:
            print("\n[客户端] 正在关闭...")

if __name__ == "__main__":
    tcp_communication()