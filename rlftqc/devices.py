import networkx as nx
import abc
from qiskit.providers.fake_provider import *
from rlftqc.simulators import CliffordGates
import cirq_google as cg

class Devices(metaclass=abc.ABCMeta):
    """Abstract class for different devices. """
    @property
    @abc.abstractmethod
    def num_qubits(self):
        """ Define the number of qubits available on the devices.
        
        Returns:
            the number of qubits in the device.
         """

    @abc.abstractmethod
    def get_gateset(self):
        """ Return the available gateset from this device. 
        
        Returns:
            List of available gates as a list of functions from CliffordGates.
            
        """

    @abc.abstractmethod
    def get_connectivity(self, qubits_index=None, directed=True):
        """Return the connectivity of the devices. 

        Args:
            qubits_index(optional, list(int)): The index of qubits to take the connectivity. Useful for taking subsets of qubits. Default: None.
            directed(optional): Whether the graph is directed or not. Set to false if the two-qubit gates are symmetric (e.g. CZ). Default: True.

        Returns:
            Edge lists of the connectivity.
        
        """

    @abc.abstractmethod
    def visualize(self):
        """ Visualize the graph of the qubit connectivity of the device. """

class IonTrap(Devices):
    """ Ion Trap devices with fully connected qubits. """

    def __init__(self, qubits=10):
        """ Initialize IonTrap device.
        
        Args:
            qubits (int): number of qubits
        """
        graph = []

        self.qubits = qubits

        for ii in range(self.qubits):
            for jj in range(ii+1, self.qubits):
                graph.append((ii,jj))
                graph.append((jj,ii))

        self.graph = nx.DiGraph()
        self.graph.add_edges_from(graph)

        self.gates = CliffordGates(self.qubits)

    @property
    def num_qubits(self):
        """ Define the number of qubits available on the devices.

        Returns:
            The number of qubits.
        """
        return self.qubits

    def get_gateset(self):
        """ Return the available gateset from this device.

        H = Rz(\pi / 2) - Rx(\pi / 2) - Rz(\pi / 2)
        S = Rz(\pi / 2) 

        Note: Ion Trap should have Molmer-Sorensen gate but it is not yet implemented it returns CX for now.

        Returns:
            List of gates from CliffordGates: [CX, H and S] gates

        """
        return [self.gates.cx, self.gates.h, self.gates.s]
     

    def get_connectivity(self, qubits_index=None, directed=True):
        """ Return the connectivity of the devices.

        Args:
            qubits_index: give qubit index to get the subset of the connectivity.
            directed: True for directed graph and False for undirected graph (in case of symmetric two-qubit gates e.g. CZ).
        
        Returns:
            The edge list
        """
        if not directed:
            graph_temp = self.graph.to_undirected()
        else:
            graph_temp = self.graph

        if qubits_index is not None:
            graph_temp = graph_temp.subgraph(qubits_index)
            
            ## relabel
            mapping = dict(zip(qubits_index, range(len(graph_temp.nodes))))
            print('Subgraph has been choosen. The node has been relabeled as follows:', mapping)

            graph_temp = nx.relabel_nodes(graph_temp, mapping)


        # Update gates
        self.gates = CliffordGates(len(qubits_index))
        self.graph = graph_temp

        return list(graph_temp.edges)

    def visualize(self):
        """ Visualize the graph of the qubit connectivity of the device. """
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos=pos, with_labels=True, node_color='w')

class IBM(Devices):
    ''' IBM Quantum devices connectivity.
    See the list of supported devices: https://docs.quantum.ibm.com/api/qiskit/0.46/providers_fake_provider#fake-v2-backends

    Args:
        name: Name of the device, see the link above.

    '''
    def __init__ (self, name):
        """ Initialize an IBM Quantum device.
        """
        try:
            self.device_ = eval("Fake%sV2()" % name.title())
            graph = list(self.device_.coupling_map)
            self.graph = nx.DiGraph()
            self.graph.add_edges_from(graph)
        except:
            try:
                self.device_ = eval("Fake%s()" % name.title())
                graph = list(self.device_.configuration().coupling_map)
                ## Add more connection for Tokyo because it is not updated by Qiskit
                if name == 'tokyo':
                    graph = graph + [(2,7), (7,2), (2,3), (3,2), (3,4), (4,3), (3,9), (9,3), (7,13), (13,7), (9,14), (14,9), (12,17), (17,12), (18,19), (19,18)]
                self.graph = nx.DiGraph()
                self.graph.add_edges_from(graph)
            except:
                print('%s name is not available in the backend. Check the full list here:  https://docs.quantum.ibm.com/api/qiskit/0.46/providers_fake_provider#fake-v2-backends' % name)

        self.gates = CliffordGates(self.num_qubits)

    @property
    def num_qubits(self):
        """ Define the number of qubits available on the devices.

        Returns:
            The number of qubits in the device.
        """
        try:
            return self.device_.num_qubits
        except:
            return self.device_.configuration().num_qubits
    
    def get_gateset(self):
        """ Return the available gateset from this device.

        S = Rz(\pi / 2)

        Returns:
            List of functions from CliffordGates: [CX, X, SQRT_X, S]
        """
        return [self.gates.cx, self.gates.x, self.gates.s, self.gates.sqrt_x]
           
    def get_connectivity(self, qubits_index=None, directed=True):
        """ Return the connectivity of the devices.

        Args:
            qubits_index: give qubit index to get the subset of the connectivity
            directed: True for directed graph and False for undirected graph (in case of symmetric two-qubit gates e.g. CZ)

        Returns:
            Edge lists of the connectivity.
        """
        if not directed:
            graph_temp = self.graph.to_undirected()
        else:
            graph_temp = self.graph

        if qubits_index is not None:
            graph_temp = graph_temp.subgraph(qubits_index)
            
            ## relabel
            mapping = dict(zip(qubits_index, range(len(graph_temp.nodes))))
            print('Subgraph has been choosen. The node has been relabeled as follows:', mapping)

            graph_temp = nx.relabel_nodes(graph_temp, mapping)


        # Update gates
        self.gates = CliffordGates(len(qubits_index))
        self.graph = graph_temp

        
        return list(graph_temp.edges)

    def visualize(self):
        """
        Visualize the graph of the qubit connectivity of the device.
        """
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos=pos, with_labels=True, node_color='w')

class Sycamore(Devices):
    ''' Google Sycamore device.
    See details: https://quantumai.google/hardware/datasheet/weber.pdf
    '''

    def __init__ (self):
        """ Initialize the Google Sycamore device. """
        device = cg.Sycamore
        self.graph = device.metadata.nx_graph
        self.graph = nx.convert_node_labels_to_integers(self.graph)
        self.graph = self.graph.to_directed()

        self.gates = CliffordGates(self.num_qubits)

    @property
    def num_qubits(self):
        """ Define the number of qubits available on the devices.
        
        Returns:
            The number of qubits.
        """
        return 54
   
    
    def get_gateset(self):
        """
        Return the available gateset from this device.

        S = U3(\pi/2, 0, \pi)
        H = U3(0,0, \pi / 2)

        Returns:
            List of functions from CliffordGates: [CZ, H, S]
        """
        return [self.gates.cz, self.gates.h, self.gates.s]

    
    def get_connectivity(self, qubits_index=None, directed=False):
        """
        Return the connectivity of the devices.
        
        Args:
            qubits_index: give qubit index to get the subset of the connectivity
            directed: True for directed graph and False for undirected graph (in case of symmetric two-qubit gates e.g. CZ)

        Returns:
            Edge lists of the connectivity.
        """
        if not directed:
            graph_temp = self.graph.to_undirected()
        else:
            graph_temp = self.graph

        if qubits_index is not None:
            graph_temp = graph_temp.subgraph(qubits_index)
            
            ## relabel
            mapping = dict(zip(qubits_index, range(len(graph_temp.nodes))))
            print('Subgraph has been choosen. The node has been relabeled as follows:', mapping)

            graph_temp = nx.relabel_nodes(graph_temp, mapping)

        
        # Update gates
        self.gates = CliffordGates(len(qubits_index))
        self.graph = graph_temp
        
        return list(graph_temp.edges)

    def visualize(self):
        """
        Visualize the graph of the qubit connectivity of the device.
        """
        pos = nx.spring_layout(self.graph, iterations=500)
        nx.draw(self.graph, pos=pos, with_labels=True, node_color='w')
