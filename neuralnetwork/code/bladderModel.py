import argparse
import time
import sys

sys.path.append('../code')
from mpi4py import MPI
from neuron import h
from matplotlib.backends.backend_pdf import PdfPages
import pickle
from tools import general_tools  as gt
import datetime
from tools import seed_handler as sh
from simulations import ForwardSimulation
from simulations import ForSimSpinalModulation
from NeuralNetwork import NeuralNetwork
from EES import EES
from BurstingEES import BurstingEES
from tools import seed_handler as sh
sh.set_seed()

comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()


class bladderModel(object):

    def __init__(self, nn_struct, stimFreq, stimAmp, sim_time, label, start_vol, end_vol):
        """
        initiate all the components needed for a bladder circuit model
         input_name: the name of the txt file that contains the model structure
         stimFreq: the stimulation frequency of EES applied on afferent
         stimAmp: the stimulation amplitude of EES applied on afferent
         sim_time: the time running the simulation
         label: the id of instance that running this simulation
         start_vol: the bladder volume at the start of simulation
         end_vol: the bladder volume at the end of simulation
        """
        self.nn_structure = nn_struct
        self.plot_mem = False # not plot anything during simulation
        self.stimFreq = stimFreq
        self.stimAmp = stimAmp
        self.simTime = sim_time # the total simulation time (ms)
        self.mnReal = True
        # self.burstingEes = None
        # self.pulsePerBurst = 5
        # self.burstFreq = 600.0
        self.seed = time.time()
        # self.memPotential = True
        self.muscleName = "Bladder"
        self.percFiberActEes = None
        # self.plotResults = True
        self.bp_pre = 0.0
        self.bp_post = 0.0
        self.bp_ratio = 0.0
        self.auc_pre = 0.0
        self.auc_post = 0.0
        self.auc_ratio = 0.0
        self.label = label
        self.start_vol = start_vol
        self.end_vol = end_vol


        if self.seed is not None:
            sh.save_seed(self.seed)
        else:
            sh.save_seed(int(time.time()))

        print("received the parameters, start initialization")


    def createNetwork(self):
        """
        create, initialize, load the nn object with predefined structure stored in txt files
        under nnStructures directory
        """

        # Create a Neuron ParallelContext object for parallel computing
        pc = h.ParallelContext()

        # initialize the neural network structure
        nn = NeuralNetwork(pc, self.nn_structure)
        # initialize the epidural electrical stimulation object
        ees = EES(pc, nn, self.stimAmp, self.stimFreq, self.simTime)

        # define patterns of stimulation, might be useful in the future...
        afferentsInput = None
        eesModulation = None

        # set cells to be recorded and their recorded types
        if self.mnReal:
            cellsToRecord = {"Pud": [nn.cells[self.muscleName]['Pud']],
                            "SPN": [nn.cells[self.muscleName]['SPN']],
                            "Pel": [nn.cells[self.muscleName]['Pel']],
                            "PMC": [nn.cells[self.muscleName]['PMC']],
                            "IN_Mn": [nn.cells[self.muscleName]['IN_Mn']],
                            "IN_Mp": [nn.cells[self.muscleName]['IN_Mp']],
                            "IN_D": [nn.cells[self.muscleName]['IN_D']],
                            "FB": [nn.cells[self.muscleName]['FB']]
                            }

            modelTypes = {
                            "Pud": "artificial",
                            "SPN": "real",
                            "Pel": "artificial",
                            "PMC": "intfire",
                            "IN_Mn": "intfire",
                            "IN_Mp": "intfire",
                            "IN_D": "intfire",
                            "FB": "intfire"
                            }
        simulation = ForSimSpinalModulation(pc, nn, cellsToRecord, modelTypes, self.stimFreq, afferentsInput, ees, eesModulation, self.simTime, self.label, self.start_vol, self.end_vol)

        # apply the EES on recruited ratio of afferent fibers
        self.percFiberActEes = ees.get_amplitude(True)
        simulation.run()

        # plot membrane potentials if recorded
        title = "Recruited Pud ratio: %.1f,Recruited Pel ratio: %.1f, Recruited SPN ratio: %.1f, Stim_Freq: %.1f Hz" % (
        self.percFiberActEes[1] * 100, self.percFiberActEes[2] * 100, self.percFiberActEes[3] * 100, self.stimFreq)
        try:
            fileName = "%.1f_pud_%.1f_pel_%.1f_spn_" % (self.percFiberActEes[1] * 100, self.percFiberActEes[2] * 100, self.percFiberActEes[3] * 100)
        except:
            print("error: can't generate filename")

        simulation.save_simulation_data(fileName, title)
        if self.plot_mem:
            simulation.plot_membrane_potatial(fileName, title)

        # record the bladder pressure pre/post stimulation, and calculate the post/pre ratio
        self.bp_pre, self.bp_post, self.bp_ratio = simulation.bladder_pressure_mean()

        # similar, calculate auc under the bladder pressure curve
        self.auc_pre, self.auc_post, self.auc_ratio = simulation.bladder_pressure_auc()

    def test_loading_network(self, nn):
        """
        helper function: test if the neural network was successfully initialized
        """
        print("primary afferents\n")
        print(nn.get_primary_afferents_names())
        print("secondary afferents\n")
        print(nn.get_secondary_afferents_names())
        print("intf_motoneuron\n")
        print(nn.get_intf_motoneurons_names())
        print("interneurons\n")
        print(nn.get_interneurons_names())
        print("get_mn_info\n")
        print(nn.get_mn_info())
