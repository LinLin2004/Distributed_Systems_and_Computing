import socket
import threading
from datetime import datetime

def tcp_daytime_server(port=6667):
    """TCP Daytime服务端"""    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  
        s.bind(('0.0.0.0', port))
        s.listen(5)
        print(f"[TCP服务端] 正在端口 {port} 监听...")
        
        try:
            while True:
                conn, addr = s.accept()
                print(f"[TCP服务端] 接受来自 {addr} 的连接")
                try:
                    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S\r\n")
                    conn.sendall(time_str.encode('ascii'))
                    print(f"[TCP服务端] 向 {addr} 发送时间")
                finally:
                    conn.close()  # 发送后立即关闭连接
        except KeyboardInterrupt:
            print("\n[TCP服务端] 关闭")

def tcp_daytime_client(server='127.0.0.1', port=6667):
    """TCP Daytime客户端"""    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(5)
        try:
            s.connect((server, port))
            data = s.recv(1024)
            print(f"[TCP客户端] 收到时间: {data.decode('ascii').strip()}")
        except ConnectionRefusedError:
            print("[TCP客户端] 错误: 连接被拒绝")
        except socket.timeout:
            print("[TCP客户端] 错误: 连接超时")
        except Exception as e:
            print(f"[TCP客户端] 错误: {e}")

if __name__ == '__main__':
    # 启动服务端线程
    server_thread = threading.Thread(target=tcp_daytime_server, daemon=True)
    server_thread.start()
    
    # 启动客户端
    tcp_daytime_client()
    
    # 等待客户端完成
    server_thread.join(timeout=5)
    print("测试完成")