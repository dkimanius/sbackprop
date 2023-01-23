import multiprocessing as mp
import time

import numpy as np

from voxelium.vae_volume.vtk_utils import initialize_vtk_resourses, make_volume_actor, numpy_volume_as_vtk_image_data, rgb_hex_to_dec


class VolumeRenderer:
    def __init__(self, queue, windowName=None, timer=100):
        self.queue = queue
        self.timer = timer
        self.nr_vols = 0
        self.volumes = None
        self.rock_ascend = True
        self.iso_min = None
        self.iso_max = None
        self.iso_steps = None
        self.iso_value = None
        self.actors = None
        self.current_actor = None
        self.current_actor_idx = 0

        self.render_window, self.renderer, self.interactor = initialize_vtk_resourses(
            windowName=windowName
        )

    def setVolumes(self, volumes):
        if volumes is None:
            return
        self.nr_vols = len(volumes)
        if self.nr_vols == 0:
            self.volumes = None
            return

        self.volumes = []
        for v in volumes:
            self.volumes.append(numpy_volume_as_vtk_image_data(v))

        self.rock_ascend = True
        self.iso_min = float(np.min(volumes[0]))
        self.iso_max = float(np.max(volumes[0]))
        self.iso_steps = float(np.std(volumes[0]) / 2.)
        if self.iso_value is None:
            self.iso_value = float(np.mean(volumes[0]) * 4.) * 4.

    def updateActors(self):
        self.removeCurrentActor()
        if self.nr_vols == 0:
            self.actors = None
        else:
            self.actors = []
            for vol in self.volumes:
                self.actors.append(make_volume_actor(vol, self.iso_value, color=rgb_hex_to_dec("c596fb")))
            if self.current_actor_idx is not None:
                self.setCurrentActor(min(self.current_actor_idx, self.nr_vols-1))
            else:
                self.setCurrentActor(0)

    def removeCurrentActor(self):
        if self.current_actor is not None:
            self.renderer.RemoveActor(self.current_actor)
            self.current_actor = None

    def setCurrentActor(self, new_actor_idx=None):
        new_actor_idx = 0 if new_actor_idx is None else new_actor_idx
        self.current_actor_idx = new_actor_idx

        if self.actors is not None \
                and self.current_actor is not None and \
                self.actors[new_actor_idx] == self.current_actor:
            return

        self.removeCurrentActor()

        if self.actors is not None:
            self.current_actor = self.actors[new_actor_idx]
            self.renderer.AddActor(self.current_actor)

        self.render_window.Render()

    def updateCurrentActorIndex(self):
        if self.nr_vols == 1:
            self.current_actor_idx = 0
            self.rock_ascend = True
            return

        self.current_actor_idx += 1 if self.rock_ascend else -1

        if self.current_actor_idx < 0:
            self.rock_ascend = True
            self.current_actor_idx = 1

        if self.current_actor_idx >= self.nr_vols:
            self.rock_ascend = False
            self.current_actor_idx = self.nr_vols - 2

    def KeyPressEvent(self, obj, _):
        if self.nr_vols == 0:
            return
        key = obj.GetKeySym()
        if key == "Up" or key == "Down":
            if key == "Up":
                self.iso_value = max(self.iso_min, self.iso_value - self.iso_steps)
            elif key == "Down":
                self.iso_value = min(self.iso_max, self.iso_value + self.iso_steps)

            self.updateCurrentActorIndex()
            self.updateActors()

        if key == "Return":
            images = []
            import matplotlib
            import vtk
            from vtk.util.numpy_support import vtk_to_numpy
            import imageio
            for i in range(self.nr_vols):
                self.setCurrentActor(i)
                vtk_win_im = vtk.vtkWindowToImageFilter()
                vtk_win_im.SetInput(self.render_window)
                vtk_win_im.Update()

                vtk_image = vtk_win_im.GetOutput()

                width, height, _ = vtk_image.GetDimensions()
                vtk_array = vtk_image.GetPointData().GetScalars()
                components = vtk_array.GetNumberOfComponents()

                arr = vtk_to_numpy(vtk_array).reshape(height, width, components)

                fn = f'dump_{i}.png'
                images.append(fn)
                arr = np.flip(arr, 0)
                matplotlib.image.imsave(fn, arr)

            with imageio.get_writer(f'dump.gif', mode='I') as writer:
                for i in range(len(images)):
                    image = imageio.imread(images[i])
                    writer.append_data(image)
                for i in range(len(images)-1):
                    image = imageio.imread(images[len(images)-i-1])
                    writer.append_data(image)

    def TimerEvent(self, obj, _):
        if not self.queue.empty():
            task = self.queue.get()
            if task is None:
                self.render_window.Finalize()
                self.interactor.TerminateApp()
                return
            self.setVolumes(task)
            self.updateActors()
        elif self.nr_vols == 0:
            self.removeCurrentActor()
            return

        if self.actors is None:
            return

        self.setCurrentActor(self.current_actor_idx)
        self.updateCurrentActorIndex()

    def start(self):
        # Wait for first volume
        while True:
            if not self.queue.empty():
                self.setVolumes(self.queue.get())
                self.updateActors()
                break
            else:
                time.sleep(0.1)

        self.interactor.Initialize()
        self.interactor.AddObserver('TimerEvent', self.TimerEvent)
        self.interactor.AddObserver("KeyPressEvent", self.KeyPressEvent)
        self.interactor.CreateRepeatingTimer(self.timer)

        self.render_window.Render()
        self.interactor.Start()
        self.render_window.Finalize()
        self.interactor.TerminateApp()

    @staticmethod
    def startNewProcess(queue):
        vr = VolumeRenderer(queue)
        vr.start()


def volumeRendererProcessLoop(volume_queue, message_queue):
    while True:
        if not message_queue.empty():
            return
        if not volume_queue.empty():
            p = mp.Process(
                target=VolumeRenderer.startNewProcess,
                args=(volume_queue,)
            )
            p.start()
            p.join()
            p.terminate()
        else:
            try:
                time.sleep(0.1)
            except KeyboardInterrupt:
                return
