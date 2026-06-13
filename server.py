import zmq
import torch
import threading
import msgpack
import numpy as np
from queue import Queue
from src.python.launcher import Launcher
from src.python.constants.server_constants import ServerConstants
import argparse

parse = argparse.ArgumentParser("")
parse.add_argument("--port", type=int, required=True)
parse.add_argument("--num-workers", type=int, default=4)
parse.add_argument("--enable-debug-mode", action="store_true")
parse.add_argument("--num_checkpoints_to_save", type = int, default=2)
parse.add_argument("--grad_acc_steps", type = int, default=1)


args = parse.parse_args()

context = zmq.Context()
frontend = context.socket(zmq.ROUTER)
frontend.bind(f"tcp://*:{args.port}")

backend = context.socket(zmq.DEALER)
backend.bind("inproc://workers")

print(f"[INFO] Server listening on port {args.port} with {args.num_workers} thread(s)")

server_config = {}
launcher = Launcher( args.num_checkpoints_to_save, args.grad_acc_steps)
debug_mode = args.enable_debug_mode

np_dtype = {
    "float": np.float32,
    "int": np.int32,
    "long int": np.longlong,
    "bool": np.bool_,
    "double": np.float64,
}

shutdown_event = threading.Event()

def worker():
    socket = context.socket(zmq.DEALER)
    socket.connect("inproc://workers")
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    while not shutdown_event.is_set():
        try:
            socks = dict(poller.poll(100))
            if socket not in socks:
                continue

            parts = socket.recv_multipart()
            client_id = parts[0]
            msg = parts[1]

            if debug_mode:
                print(f"[DEBUG] Received parts: {len(parts)} from client {client_id}")

            unpacked_data = msgpack.unpackb(msg, raw=False)

            tensors = []
            if len(parts) > 2:
                for tensor_part in parts[2:]:
                    try:
                        spec = msgpack.unpackb(tensor_part, raw=False)
                        shape = tuple(spec[0])
                        dtype = np_dtype[spec[1]]
                        data = np.frombuffer(spec[2], dtype=dtype).copy()
                        tensors.append(torch.from_numpy(data).reshape(shape))
                    except Exception as e:
                        if debug_mode:
                            print(f"[ERROR] Tensor unpack failed: {e}")
                        raise RuntimeError()

            func_name = unpacked_data.get(ServerConstants.FUNC, "")
            print(func_name)
            if func_name == "finish_server":
                print("[INFO] Shutdown command received.")
                shutdown_event.set()
                result = {"status": "server_shutting_down"}
                socket.send_multipart([client_id, msgpack.packb(result, use_bin_type=True)])
                break

            try:
                launcher_func = getattr(launcher, func_name)
                if tensors:
                    result = launcher_func(server_config, unpacked_data, tensors)
                else:
                    result = launcher_func(server_config, unpacked_data)
            except Exception as e:
                if debug_mode:
                    import traceback
                    traceback.print_exc()
                result = {"error": str(e)}

            bin_msg = msgpack.packb(result, use_bin_type=True)
            socket.send_multipart([client_id, bin_msg])

        except zmq.ContextTerminated:
            break 
        except Exception as e:
            if debug_mode:
                print(f"[ERROR] Exception in worker: {e}")
            raise RuntimeError(f"[ERROR] Exception in worker: {e}")
            # continue

    socket.close()



def listener():
    poller = zmq.Poller()
    poller.register(frontend, zmq.POLLIN)
    poller.register(backend, zmq.POLLIN)

    while not shutdown_event.is_set():
        try:
            socks = dict(poller.poll(timeout=100))
        except zmq.ZMQError:
            break

        if frontend in socks:
            msg = frontend.recv_multipart()
            backend.send_multipart(msg)

        if backend in socks:
            msg = backend.recv_multipart()
            frontend.send_multipart(msg)

worker_threads = []
for _ in range(args.num_workers):
    thread = threading.Thread(target=worker)
    thread.start()
    worker_threads.append(thread)

try:
    listener()
except KeyboardInterrupt:
    print("[INFO] Manual shutdown triggered.")
    shutdown_event.set()

for thread in worker_threads:
    thread.join(timeout=100)

frontend.close(0)
backend.close(0)
context.term()

print("[INFO] Server shutdown complete.")
