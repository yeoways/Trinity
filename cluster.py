class TrinityControllerTest(object):

    def __init__(self, num_gpus, num_cpus=1, start_virtual_GPU=True, allow_soft_placement=False, disable_detailed_stats=True, disable_timeline=True):
        self.cluster = self._buildCluster(num_cpus, num_gpus, allow_soft_placement, disable_detailed_stats, disable_timeline, start_virtual_GPU)
        pass

    @staticmethod
    def _buildCluster(num_cpus=0, num_gpus=1, allow_soft_placement=True, disable_detailed_stats=True, disable_timeline=True, start_virtual_GPU=True):
        devices = []
        # 配置GPU
        if num_gpus > 0:
            device_properties = device_properties_pb2.DeviceProperties(
                type='GPU',
                vendor='NVidia',
                model='GeForce GTX TITAN X',
                frequency=1076,
                num_cores=24,
                environment={'architecture': '5.2',
                             'cuda': '8000',
                             'cudnn': '6021'},
                num_registers=65536,
                l1_cache_size=24576,
                l2_cache_size=3145728,
                shared_memory_size_per_multiprocessor=98304,
                memory_size=12783648768,
                bandwidth=336480000)
            for i in range(num_gpus):
                devices.append(
                    device_properties_pb2.NamedDevice(
                        properties=device_properties, name='/GPU:' + str(i)))
        # 配置CPU
        if num_cpus > 0:
            device_properties = device_properties_pb2.DeviceProperties(
                type='CPU',
                frequency=2000,
                num_cores=4,
                l1_cache_size=32768,
                l2_cache_size=262144,
                l3_cache_size=12582912)
            for i in range(num_cpus):
                devices.append(
                    device_properties_pb2.NamedDevice(
                        properties=device_properties, name='/CPU:' + str(i)))
        if start_virtual_GPU:
            return cluster.Cluster(devices=devices)
        else:
            return cluster.Cluster(allow_soft_placement, disable_detailed_stats, disable_timeline)

    def getCluster(self):
        return self.cluster
