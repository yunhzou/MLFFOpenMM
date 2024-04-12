import openmm as mm
from openmm import app, unit
import numpy as np

def create_water_topology():
    # Initialize the topology
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("water", chain)
    element_O = app.Element.getByAtomicNumber(8)
    element_H = app.Element.getByAtomicNumber(1)
    atom0 = topology.addAtom("O", element_O, residue)
    atom1 = topology.addAtom("H", element_H, residue)
    atom2 = topology.addAtom("H", element_H, residue)
    topology.addBond(atom0, atom1)
    topology.addBond(atom0, atom2)
    return topology

def setup_simulation(topology):
    # Load the forcefield for water
    forcefield = app.ForceField("amber14/tip3pfb.xml")

    # Create the system based on the topology with a specified cutoff for nonbonded interactions
    system = forcefield.createSystem(topology, nonbondedCutoff=1 * unit.nanometer)
    
    # Define an integrator
    integrator = mm.VerletIntegrator(2 * unit.femtoseconds)
    
    # Initialize the simulation
    simulation = app.Simulation(topology, system, integrator)
    return simulation

def generate_configuration(num_atoms):
    # Generate random configurations within a 2x2x2 Å box centered at the origin
    return np.random.rand(num_atoms, 3) * 2 - 1

def generate_configurations(num_atoms, num_configs):
    """
    Generates random 3D configurations for a specified number of atoms across multiple configurations.
    
    Args:
    num_atoms (int): Number of atoms per configuration.
    num_configs (int): Number of configurations to generate.

    Returns:
    np.ndarray: Array of shape (num_configs, num_atoms, 3) containing the configurations.
    """
    # Each dimension will have coordinates in the range [-1, 1) multiplied by the box size, centered at the origin
    return np.random.rand(num_configs, num_atoms, 3) * 2 - 1

def calculate_potential_energy_and_forces(simulation, configuration):
    # Set the positions of the atoms in the simulation
    simulation.context.setPositions(configuration * unit.angstroms)
    
    # Get the state of the simulation including the potential energy
    state = simulation.context.getState(getEnergy=True,getForces=True)
    pe = state.getPotentialEnergy()
    forces = state.getForces(asNumpy=True)
    return pe, forces

# Main execution
if __name__ == '__main__':
    num_atoms = 3  # For a water molecule (H2O)
    num_configs = 5000  # Number of configurations
    configurations = generate_configurations(num_atoms, num_configs)
    
    # Create the topology and set up the simulation
    topology = create_water_topology()
    simulation = setup_simulation(topology)
    
    # Initialize an array to store the potential energies
    potential_energies = np.zeros(num_configs)
    forces = np.zeros((num_configs, num_atoms, 3))

    # Calculate potential energy for each configuration
    for i, config in enumerate(configurations):
        potential_energy, force = calculate_potential_energy_and_forces(simulation, config)
        potential_energies[i] = potential_energy.value_in_unit(unit.kilojoules_per_mole)
        forces[i] = force.value_in_unit(unit.kilojoules_per_mole/unit.nanometer)

    

    # Save configurations and potential energies to an NPZ file
    np.savez(r'dataset\water_configurations_and_potentials.npz', configurations=configurations, potentials=potential_energies, forces=forces)

    print("Data has been saved to "+ r'dataset\water_configurations_and_potentials.npz')
