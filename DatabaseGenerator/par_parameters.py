# Created by Applied Geometry Laboratory (https://github.com/appliedgeometry)
# for "EuLearn: A 3D database for learning Euler characteristics" Project, 2025.
#
#
# https://huggingface.co/datasets/appliedgeometry/EuLearn
# https://github.com/appliedgeometry/EuLearn_db
#
# GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--o', '--out_dir', type=str,
                    default='./output', help='Output directory', metavar='o')

parser.add_argument('--mvx', '--min_voxels', type=int,
                    default=10, help='min voxels per axis', metavar='mvx')

parser.add_argument('--pr', '--progress', type=str, help='show progress bar', metavar='pr')

parser.add_argument('--r_neigh', '--r_neighborhood', type=float,
                    default=0.07, help='Max neighborhood to blow up', metavar='r_neigh')

parser.add_argument('--n', '--k_nodes', type=int,
                    default=100, help='nodes in knot', metavar='n')

parser.add_argument('--name_format', '--nfmt', type=str,
                    default='type,parameters,Nbars,Nvoxels', help='name format', metavar='nfmt')

parser.add_argument('--arclength_tolerance', '--alt', type=float,
                    default=0.01,help='arc length tolerance', metavar='alt')

parser.add_argument('--reachrepo', '--reachrepo', type=str,
                    default='./../Alcances',help='Reach repository scripts', metavar='reachrepo')

parser.add_argument('--auto_submethod', '--asubm', type=str,
                    default='reach',help='submethod for automatic blow up', metavar='auto_submethod')

parser.add_argument('--vx_ActDistSc','--vads',type=float,
                    default=1.732050807569,help='Voxel Activation Distance Scale', metavar='vads')

parser.add_argument('--m', '--method', type=str, help='pump method', metavar='m')

parser.add_argument('--kf', '--knots_file', type=str, help='knot file', metavar='k')

parser.add_argument('--vtx_split', '--vtx_split', type=int, help='Split STL', metavar='vtx')

parser.add_argument('--log', '--log_file', type=str, help='LOG file', metavar='log')

parser.add_argument('--bboffset','--bboffset', type=int, help='bounding box offset in voxels', metavar='bboffset')

parser.add_argument('--bpb','--bpb', type=int, help='Max blocks per batch', metavar='bpb')

parser.add_argument('--ascii_names','--asc', type=str, help='arcii chars in filenames', metavar='ascii_names')

parser.add_argument('--fxnormals', '--fxn',type=str,help='fix normals directory',metavar='fxn')

parser.add_argument('--copy_from', '--cpf',type=str,help='copy_from directory to rename STLS classificator',metavar='cpf')

parser.add_argument('--fields_dir', '--flds',type=str,help='scalar fields directory',metavar='flds')
parser.add_argument('--nsmooth_dir', '--nsmth',type=str,help='non-smooth STL directory',metavar='nsmth')
parser.add_argument('--bup_dir', '--bup',type=str,help='blow-up values - directory',metavar='bup')

parser.add_argument('--reach_threshold', '--rchthrsld', type=str,help='reach threshold', metavar='rth')

parser.add_argument('--reach_iterations', '--rchit', type=str,help='reach iterations', metavar='rit')

# sinusoidal method
parser.add_argument('--sine_const', '--sn_cte', type=float,help='height shift position (sine)', metavar='shift')
parser.add_argument('--sine_amp', '--sn_amp', type=float,help='sine amplitude', metavar='amp')
parser.add_argument('--sine_freq', '--sn_fq', type=float,help='sine frequency', metavar='freq')
parser.add_argument('--sine_phase', '--sn_ph', type=str,help='sine phase (str)', metavar='ph')

#

args = parser.parse_args()

#print(args)