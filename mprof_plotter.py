import glob
import os
import os.path as osp
import sys
import re
import copy
import time
import math
import logging
import itertools
from ast import literal_eval
from matplotlib.patches import Rectangle

from collections import defaultdict
from argparse import ArgumentParser, ArgumentError, REMAINDER, RawTextHelpFormatter

def add_brackets(xloc, yloc, xshift=0, color="r", label=None, options=None):
    """Add two brackets on the memory line plot.

    This function uses the current figure.

    Parameters
    ==========
    xloc: tuple with 2 values
        brackets location (on horizontal axis).
    yloc: tuple with 2 values
        brackets location (on vertical axis)
    xshift: float
        value to subtract to xloc.
    """
    try:
        import pylab as pl
    except ImportError as e:
        print("matplotlib is needed for plotting.")
        print(e)
        sys.exit(1)
    height_ratio = 20.
    vsize = (pl.ylim()[1] - pl.ylim()[0]) / height_ratio
    hsize = (pl.xlim()[1] - pl.xlim()[0]) / (10. * height_ratio)

    bracket_x = pl.asarray([hsize, 0, 0, hsize])
    bracket_y = pl.asarray([vsize, vsize, -vsize, -vsize])

    # Matplotlib workaround: labels starting with _ aren't displayed
    if label[0] == '_':
        label = ' ' + label

    # pl.plot(bracket_x + xloc[0] - xshift, bracket_y + yloc[0],
    #             "-" + color, linewidth=2, label=label)
    # pl.plot(-bracket_x + xloc[1] - xshift, bracket_y + yloc[1],
    #             "-" + color, linewidth=2)
    pl.plot(xloc[0] - xshift, yloc[0], color=color, marker='o', ms=3, alpha=0.5, label=label)
    pl.plot(xloc[1] - xshift, yloc[1], color=color, marker='o', ms=3, alpha=0.5)

    
    ax = pl.gca()
    yl = ax.get_ylim()
    ax.add_patch(Rectangle((xloc[0]-xshift, yl[0]),
                        xloc[1]-xloc[0], yl[1] - yl[0],
                        fc = color, 
                        ec ='none',
                        alpha=0.2))
    ax.set_ylim(yl)

        # TODO: use matplotlib.patches.Polygon to draw a colored background for
        # each function.

        # with maplotlib 1.2, use matplotlib.path.Path to create proper markers
        # see http://matplotlib.org/examples/pylab_examples/marker_path.html
        # This works with matplotlib 0.99.1
        ## pl.plot(xloc[0], yloc[0], "<"+color, markersize=7, label=label)
        ## pl.plot(xloc[1], yloc[1], ">"+color, markersize=7)


def read_mprofile_file(filename):
    """Read an mprofile file and return its content.

    Returns
    =======
    content: dict
        Keys:

        - "mem_usage": (list) memory usage values, in MiB
        - "timestamp": (list) time instant for each memory usage value, in
            second
        - "func_timestamp": (dict) for each function, timestamps and memory
            usage upon entering and exiting.
        - 'cmd_line': (str) command-line ran for this profile.
    """
    func_ts = {}
    mem_usage = []
    timestamp = []
    children  = defaultdict(list)
    cmd_line = None
    f = open(filename, "r")
    for l in f:
        if l == '\n':
            raise ValueError('Sampling time was too short')
        field, value = l.split(' ', 1)
        if field == "MEM":
            # mem, timestamp
            values = value.split(' ')
            mem_usage.append(float(values[0]))
            timestamp.append(float(values[1]))

        elif field == "FUNC":
            values = value.split(' ')
            f_name, mem_start, start, mem_end, end = values[:5]
            ts = func_ts.get(f_name, [])
            to_append = [float(start), float(end), float(mem_start), float(mem_end)]
            if len(values) >= 6:
                # There is a stack level field
                stack_level = values[5]
                to_append.append(int(stack_level))
            ts.append(to_append)
            func_ts[f_name] = ts

        elif field == "CHLD":
            values = value.split(' ')
            chldnum = values[0]
            children[chldnum].append(
                (float(values[1]), float(values[2]))
            )

        elif field == "CMDLINE":
            cmd_line = value
        else:
            pass
    f.close()

    return {"mem_usage": mem_usage, "timestamp": timestamp,
            "func_timestamp": func_ts, 'filename': filename,
            'cmd_line': cmd_line, 'children': children}


def plot_file(filename, index=0, timestamps=True, children=True, options=None):
    try:
        import pylab as pl
    except ImportError as e:
        print("matplotlib is needed for plotting.")
        print(e)
        sys.exit(1)
    import numpy as np  # pylab requires numpy anyway
    mprofile = read_mprofile_file(filename)

    if len(mprofile['timestamp']) == 0:
        print('** No memory usage values have been found in the profile '
              'file.**\nFile path: {0}\n'
              'File may be empty or invalid.\n'
              'It can be deleted with "mprof rm {0}"'.format(
            mprofile['filename']))
        sys.exit(0)

    # Merge function timestamps and memory usage together
    ts = mprofile['func_timestamp']
    t = mprofile['timestamp']
    mem = mprofile['mem_usage']
    chld = mprofile['children']

    if len(ts) > 0:
        for values in ts.values():
            for v in values:
                t.extend(v[:2])
                mem.extend(v[2:4])

    mem = np.asarray(mem)
    t = np.asarray(t)
    ind = t.argsort()
    mem = mem[ind]
    t = t[ind]

    # Plot curves
    fig = pl.figure(figsize=[9,5])
    global_start = float(t[0])
    t = t - global_start

    max_mem = mem.max()
    max_mem_ind = mem.argmax()

    all_colors = ("c", "y", "g", "r", "b")
    mem_line_colors = ("k", "b", "r", "g", "c", "y", "m")

    show_trend_slope = options is not None and hasattr(options, 'slope') and options.slope is True

    mem_line_label = time.strftime("%d / %m / %Y - start at %H:%M:%S",
                                   time.localtime(global_start)) \
                     + ".{0:03d}".format(int(round(math.modf(global_start)[0] * 1000)))

    mem_trend = None
    if show_trend_slope:
        # Compute trend line
        mem_trend = np.polyfit(t, mem, 1)

        # Append slope to label
        mem_line_label = mem_line_label + " slope {0:.5f}".format(mem_trend[0])

    pl.plot(t, mem, "-" + mem_line_colors[index % len(mem_line_colors)],
            label=mem_line_label)

    if show_trend_slope:
        # Plot the trend line
        pl.plot(t, t*mem_trend[0] + mem_trend[1], "--", linewidth=0.5, color="#00e3d8")

    bottom, top = pl.ylim()
    bottom += 0.001
    top -= 0.001

    # plot children, if any
    if len(chld) > 0 and children:
        cmpoint = (0,0) # maximal child memory

        for idx, (proc, data) in enumerate(chld.items()):
            # Create the numpy arrays from the series data
            cts  = np.asarray([item[1] for item in data]) - global_start
            cmem = np.asarray([item[0] for item in data])

            cmem_trend = None
            child_mem_trend_label = ""
            if show_trend_slope:
                # Compute trend line
                cmem_trend = np.polyfit(cts, cmem, 1)

                child_mem_trend_label = " slope {0:.5f}".format(cmem_trend[0])

            # Plot the line to the figure
            pl.plot(cts, cmem, "-" + mem_line_colors[(idx + 1) % len(mem_line_colors)],
                    label="child {}{}".format(proc, child_mem_trend_label))

            if show_trend_slope:
                # Plot the trend line
                pl.plot(cts, cts*cmem_trend[0] + cmem_trend[1], "--", linewidth=0.5, color="black")

            # Detect the maximal child memory point
            cmax_mem = cmem.max()
            if cmax_mem > cmpoint[1]:
                cmpoint = (cts[cmem.argmax()], cmax_mem)

        # Add the marker lines for the maximal child memory usage
        pl.vlines(cmpoint[0], pl.ylim()[0]+0.001, pl.ylim()[1] - 0.001, 'r', '--')
        pl.hlines(cmpoint[1], pl.xlim()[0]+0.001, pl.xlim()[1] - 0.001, 'r', '--')

    # plot timestamps, if any
    if len(ts) > 0 and timestamps:
        func_num = 0
        f_labels = function_labels(ts.keys())
        for f, exec_ts in ts.items():
            for execution in exec_ts:
                add_brackets(execution[:2], execution[2:], xshift=global_start,
                             color=all_colors[func_num % len(all_colors)],
                             label=f_labels[f]
                                   + " %.3fs" % (execution[1] - execution[0]), options=options)
            func_num += 1

    if timestamps:
        pl.hlines(max_mem,
                  pl.xlim()[0] + 0.001, pl.xlim()[1] - 0.001,
                  colors="r", linestyles="--")
        pl.vlines(t[max_mem_ind], bottom, top,
                  colors="r", linestyles="--")

    leg = pl.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=5, ncol=2)
    leg.get_frame().set_alpha(0.5)
    pl.grid()
    pl.xlabel("time (in seconds)")
    pl.ylabel("memory used (in MiB)")
    pl.tight_layout()
    return mprofile

def add_timestamp_rectangle(ax, x0, x1, y0, y1, func_name, color='none'):
    rect = ax.fill_betweenx((y0, y1), x0, x1, color=color, alpha=0.5, linewidth=1)
    text = ax.text(x0, y1, func_name,
        horizontalalignment='left',
        verticalalignment='top',
        color=(0, 0, 0, 0)
    )
    return rect, text


def function_labels(dotted_function_names):
    state = {}

    def set_state_for(function_names, level):
        for fn in function_names:
            label = ".".join(fn.split(".")[-level:])
            label_state = state.setdefault(label, {"functions": [],
                                                   "level": level})
            label_state["functions"].append(fn)

    set_state_for(dotted_function_names, 1)

    while True:
        ambiguous_labels = [label for label in state if len(state[label]["functions"]) > 1]
        for ambiguous_label in ambiguous_labels:
            function_names = state[ambiguous_label]["functions"]
            new_level = state[ambiguous_label]["level"] + 1
            del state[ambiguous_label]
            set_state_for(function_names, new_level)
        if len(ambiguous_labels) == 0:
            break

    fn_to_label = dict((label_state["functions"][0] , label) for label, label_state in state.items())

    return fn_to_label


def plot_action():
    def xlim_type(value):
        try:
            newvalue = [float(x) for x in value.split(',')]
        except:
            raise ArgumentError("'%s' option must contain two numbers separated with a comma" % value)
        if len(newvalue) != 2:
            raise ArgumentError("'%s' option must contain two numbers separated with a comma" % value)
        return newvalue

    desc = """Plots using matplotlib the data file `file.dat` generated
using `mprof run`. If no .dat file is given, it will take the most recent
such file in the current directory."""
    parser = ArgumentParser(usage="mprof plot [options] [file.dat]", description=desc)
    parser.add_argument('--version', action='version', version=mp.__version__)
    parser.add_argument("--title", "-t", dest="title", default=None,
                        type=str, action="store",
                        help="String shown as plot title")
    parser.add_argument("--no-function-ts", "-n", dest="no_timestamps", action="store_true",
                        help="Do not display function timestamps on plot.")
    parser.add_argument("--output", "-o",
                        help="Save plot to file instead of displaying it.")
    parser.add_argument("--window", "-w", dest="xlim", type=xlim_type,
                        help="Plot a time-subset of the data. E.g. to plot between 0 and 20.5 seconds: --window 0,20.5")
    parser.add_argument("--flame", "-f", dest="flame_mode", action="store_true",
                        help="Plot the timestamps as a flame-graph instead of the default brackets")
    parser.add_argument("--slope", "-s", dest="slope", action="store_true",
                        help="Plot a trend line and its numerical slope")
    parser.add_argument("--backend",
                      help="Specify the Matplotlib backend to use")
    parser.add_argument("profiles", nargs="*",
                        help="profiles made by mprof run")
    args = parser.parse_args()

    try:
        if args.backend is not None:
            import matplotlib
            matplotlib.use(args.backend)

        import pylab as pl
    except ImportError as e:
        print("matplotlib is needed for plotting.")
        print(e)
        sys.exit(1)
    pl.ioff()

    filenames = get_profiles(args)

    fig = pl.figure(figsize=(14, 6), dpi=90)
    if not args.flame_mode:
        ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    else:
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    if args.xlim is not None:
        pl.xlim(args.xlim[0], args.xlim[1])

    if len(filenames) > 1 or args.no_timestamps:
        timestamps = False
    else:
        timestamps = True
    plotter = plot_file
    if args.flame_mode:
        plotter = flame_plotter
    for n, filename in enumerate(filenames):
        mprofile = plotter(filename, index=n, timestamps=timestamps, options=args)
    pl.xlabel("time (in seconds)")
    pl.ylabel("memory used (in MiB)")

    if args.title is None and len(filenames) == 1:
        pl.title(mprofile['cmd_line'])
    else:
        if args.title is not None:
            pl.title(args.title)

    # place legend within the plot, make partially transparent in
    # case it obscures part of the lineplot
    if not args.flame_mode:
        leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        leg.get_frame().set_alpha(0.5)
        pl.grid()

    if args.output:
        pl.savefig(args.output)
    else:
        pl.show()

def filter_mprofile_mem_usage_by_function(prof, func):
    if func is None:
        return prof["mem_usage"]

    if func not in prof["func_timestamp"]:
        raise ValueError(str(func) + " was not found.")

    time_ranges = prof["func_timestamp"][func]
    filtered_memory = []
    
    # The check here could be improved, but it's done in this
    # inefficient way to make sure we don't miss overlapping
    # ranges.
    for mib, ts in zip(prof["mem_usage"], prof["timestamp"]):
        for rng in time_ranges:
            if rng[0] <= ts <= rng[1]:
                filtered_memory.append(mib)

    return filtered_memory

def peak_action():
    desc = """Prints the peak memory used in data file `file.dat` generated
using `mprof run`. If no .dat file is given, it will take the most recent
such file in the current directory."""
    parser = ArgumentParser(usage="mprof peak [options] [file.dat]", description=desc)
    parser.add_argument("profiles", nargs="*",
                    help="profiles made by mprof run")
    parser.add_argument("--func", dest="func", default=None,
                        help="""Show the peak for this function. Does not support child processes.""")
    args = parser.parse_args()
    filenames = get_profiles(args)

    for filename in filenames:
        prof = read_mprofile_file(filename)
        try:
            mem_usage = filter_mprofile_mem_usage_by_function(prof, args.func)
        except ValueError:
            print("{}\tNaN MiB".format(prof["filename"]))
            continue

        print("{}\t{:.3f} MiB".format(prof["filename"], max(mem_usage)))
        for child, values in prof["children"].items():
            child_peak = max([ mem_ts[0] for mem_ts in values ])
            print("  Child {}\t\t\t{:.3f} MiB".format(child, child_peak))
        

def get_profiles(args):
    profiles = glob.glob("mprofile_??????????????.dat")
    profiles.sort()

    if len(args.profiles) == 0:
        if len(profiles) == 0:
            print("No input file found. \nThis program looks for "
                  "mprofile_*.dat files, generated by the "
                  "'mprof run' command.")
            sys.exit(-1)
        print("Using last profile data.")
        filenames = [profiles[-1]]
    else:
        filenames = []
        for prof in args.profiles:
            if osp.exists(prof):
                if not prof in filenames:
                    filenames.append(prof)
            else:
                try:
                    n = int(prof)
                    if not profiles[n] in filenames:
                        filenames.append(profiles[n])
                except ValueError:
                    print("Input file not found: " + prof)
    if not len(filenames):
        print("No files found from given input.")
        sys.exit(-1)

    return filenames