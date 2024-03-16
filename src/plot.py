#!/usr/bin/env python
import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import peak_widths,find_peaks,argrelmax,correlate
from scipy.optimize import curve_fit

fps = 240
(ball_width_pixels, ball_width_metres) = (50, 0.02217)
pixel_scale_factor = ball_width_metres / ball_width_pixels
ball_mass = 0.044756

def parse_json():
    command_arguments = sys.argv[1:]

    filename = ""
    file_ext = "-data.json"
    if len(command_arguments) > 0:
        (filename, _) = os.path.splitext(command_arguments[0])
        (dirs, basename) = os.path.split(filename)

        dirs = dirs.split('/')
        if dirs[0].lower() == "res":
            dirs[0] = "data"
            dirs.append(basename)
            filename = '/'.join(dirs)
    else:
        print("Please enter a filename")
        return

    with open(f"{filename}{file_ext}") as json_data:
        data = json.load(json_data)    

    position_json_data = np.array(data["positions"])
    stddev_json_data = np.array(data["stddevs"])

    return (position_json_data, stddev_json_data, filename)

def colour_from_index(i):
    match i:
        case 0:
            rgb = (0,255,0)
        case 4:
            rgb = (0,255,255)

        case 1:
            rgb = (255,0,0)
        case 3:
            rgb = (255,0,255)

        case 2:
            rgb = (255,255,0)
        case _:
            rgb = (0,0,0)

    return "#{r:02X}{g:02X}{b:02X}".format(r = rgb[0], g = rgb[1], b = rgb[2])   

def get_time_period(positions, plot, transform):
    (width, prominence) = (15, 0.004)
    (peak_indices, info) = find_peaks(-positions[0], width=width, prominence=prominence)
    (left_ips, right_ips) = (info.get("left_ips"), info.get("right_ips"))
    consecutive_diff = np.abs(peak_indices[:len(peak_indices) - 1] - peak_indices[1:len(peak_indices)])
    filtered_diff = consecutive_diff[np.where(consecutive_diff > 10)]
    period = np.sum(filtered_diff) / (len(filtered_diff) * fps)

    plot.text(len(positions[0])/fps * 0.9, min(positions[0]) * 0.9, f"$T_0 = {{{period:.3f}}}\,s$", fontsize='x-large')

def get_peak_indices(position):
    epsilon = 120
    width  = 15
    height_0 = max(position) * 0.1

    (peak_indices_0, info_0) = find_peaks(position, width=width, height=height_0, prominence=0.001)
    peak_heights_0 = info_0["peak_heights"]

    consecutive_diff_x_0 = np.abs(peak_indices_0[:-1] - peak_indices_0[1:])

    filtered_indices_0 = []
    filtered_heights_0 = []
    for (loop_index, (i1, i2)) in enumerate(zip(peak_indices_0[:-1], peak_indices_0[1:])):
        if abs(i1 - i2) > width * 0.9:
            filtered_indices_0.append(peak_indices_0[loop_index])
            filtered_heights_0.append(peak_heights_0[loop_index])

    return (filtered_indices_0, filtered_heights_0)

def exp_func(x, a, k):
    return a * np.exp(k*x)

def find_amplitude_loss(t, y_data, plt, ball_mass):
    epsilon = 40
    cutoff_index = max(np.argmax(y_data) - epsilon, 0)

    y_cutoff = y_data[cutoff_index:]
    
    (width, height) = (20, max(y_cutoff) * 0.1)
    (peak_indices, info) = find_peaks(y_cutoff, width=width,height=height)
    peak_heights = info["peak_heights"]

    consecutive_time_diff = peak_indices[1:len(peak_indices)] - peak_indices[:len(peak_indices) - 1]
    filtered_indices = np.where(consecutive_time_diff > width * 0.9)
    filtered_heights = peak_heights[filtered_indices]

    t_0 = np.linspace(t[0]+cutoff_index / fps, t[-1], num=len(filtered_heights))
    # (param, _) = curve_fit(exp_func, t_0, filtered_heights, p0=(1,-1))

    # for index in peak_indices[filtered_indices]:
    #     plt.axvline(x=(index+cutoff_index)/fps)

    # plt.text(max(t) * 0.8, max(y_cutoff) * 0.7, "{:f}".format(param[1] * 2 * ball_mass))
    # plt.plot(t_0, exp_func(t_0, *param), '-', label=r"$y = {:.2f}e^{{{:.2f}t}}$".format(*param), color='r', lw=1)

    param = [1,1]
    
    return (exp_func(t, *param), param)

def damped_sine(t, A, b, omega, phi):
    return A * np.cos(omega*t - phi) * np.exp(b * t) 

def plot_data(positions, stddev, filename):
    conv_width = 10
    time_step = 2.5
    
    x_positions = [np.convolve(pos[:,1], np.ones(conv_width), 'valid') / conv_width for pos in positions]

    # Font Sizes
    fontsize = 14
    plt.rc('font', size=fontsize)
    plt.rc('axes', titlesize=fontsize)
    plt.rc('axes', labelsize=fontsize)
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.rc('legend', fontsize=fontsize)

    (fig, subplot) = plt.subplots(1,1)
    fig.suptitle(f"Analysis of {filename}", x=0.518, y=0.98 , horizontalalignment="center")
    fig.set_size_inches((1920 / 100, 1080 / 100), forward=True)
    fig.set_dpi(120)
    # plt.figlegend(ncols=len(x_positions) / 2, bbox_to_anchor=(1, 0.5), markerscale=10000, alignment="center", framealpha=1)

    epsilon = ball_width_pixels * 0.5
    mean_positions = [0] * 5
    for (i, pos) in enumerate(x_positions):
        if i > (len(x_positions) - 1) / 2:
            part_of_pos = pos[np.where(pos < min(pos) + epsilon)]
        elif i == (len(x_positions) - 1) / 2:
            part_of_pos = pos
        else:
            part_of_pos = pos[np.where(pos > max(pos) - epsilon)]

        mean_positions[i] = sum(part_of_pos)/len(part_of_pos)

    equilibrium_positions = [(pos - mean_positions[i]) * pixel_scale_factor for (i, pos) in enumerate(x_positions)]

    (peak_indices, peak_heights) = get_peak_indices(-equilibrium_positions[0])
    index_at_release = peak_indices[0]

    t = np.linspace(0, len(equilibrium_positions[0]) / fps, num=len(peak_indices))
    # (param, _) = curve_fit(exp_func, t, -np.array(peak_heights), p0=(1,-1))
    # subplot.plot(t, exp_func(t, *param), '-', label=r"$y = {:.2f}e^{{{:.2f}t}}$".format(*param), color='r', lw=1)

    get_time_period(equilibrium_positions, subplot, subplot.bbox.transformed(plt.gca().transAxes))

    # (peak_indices, peak_heights) = get_peak_indices(equilibrium_positions[-1])
    # t = np.linspace(t[0], t[-1], num=len(peak_indices))
    # (param, _) = curve_fit(exp_func, t, peak_heights, p0=(1,-1))
    # subplot.plot(t, exp_func(t, *param), '-', label=r"$y = {:.2f}e^{{{:.2f}t}}$".format(*param), color='r', lw=1)

    # # Fit the motion to damped SHO
    # filtered_pos = equilibrium_positions[-1][index_at_release:]
    # t = np.linspace(t[0], t[-1], num=len(filtered_pos)) 
    # (param, _) = curve_fit(damped_sine, t, filtered_pos, p0=(4, -0.01, 10, 0))

    # subplot.plot(t, damped_sine(t, *param), '-', color='r', lw=1, label=r"$y = {0:.2f}cos({2:.2f}t - {1:.2f}) e^{{{3:.2f}t}}$".format(*param))
    # subplot.plot(t, filtered_pos, '-', label="Ball {i} Position", color='b', lw=1)

    t = np.linspace(t[0], t[-1], num=len(equilibrium_positions[0])) 
    smoothed_stddev = [(np.convolve(std * pixel_scale_factor, np.ones(conv_width), 'valid') / conv_width) for std in stddev]
    # smoothed_stddev = [(np.convolve(std, np.ones(conv_width), 'valid') / conv_width) for std in stddev]

    energies = [0.5 * ball_mass * ((np.diff(pos)/np.diff(t)) ** 2) for pos in equilibrium_positions]
    total_energy = np.convolve(np.sum(energies, axis=0), np.ones(conv_width), 'valid') / conv_width

    # freq = np.fft.fftfreq(len(x_positions[0])).real
    # ffts = [np.fft.fft(positions).real for positions in equilibrium_positions]

    for i in range(0, len(equilibrium_positions)):
        col = colour_from_index(i)

        ## Absolute positions of the balls
        # subplot.plot(t, x_positions[i], '-', label="Ball {i} Position".format(i=i+1), color=col, lw=1)
        # subplot.fill_between(t, x_positions[i] - smoothed_stddev[i], x_positions[i] + smoothed_stddev[i], color=col, alpha=0.25)
        
        subplot.plot(t, equilibrium_positions[i], '-', label="Ball {i} Displacement".format(i=i+1), color=col, lw=1)
        subplot.fill_between(t, equilibrium_positions[i] - smoothed_stddev[i], equilibrium_positions[i] + smoothed_stddev[i], color=col, alpha=0.25)

    subplot.set_title("$x$-Displacement Over Time")
    subplot.set_xlabel("$t$ [s]")
    subplot.set_ylabel("$x$-Displacement [m]")
    subplot.legend(ncols=len(equilibrium_positions))
    subplot.margins(x=0.01, y=0.01, tight=True)
    subplot.set_xticks(np.arange(t[0], t[-1]+time_step, step=time_step))

    fig.tight_layout(pad=2.0)
    fig.savefig(f"{filename}-positions.pdf")

    plt.figure(1)
    (fig, subplot) = plt.subplots(1,1)
    fig.suptitle(f"Analysis of {filename}", x=0.518, y=0.98 , horizontalalignment="center")
    fig.set_size_inches((1920 / 100, 1080 / 100), forward=True)
    fig.set_dpi(120)
    # plt.figlegend(ncols=len(x_positions) / 2, bbox_to_anchor=(1, 0.5), markerscale=10000, alignment="center", framealpha=1)

    # for i in range(0, len(energies)):
    #     col = colour_from_index(i)

    #     subplot.plot(t, energies[i], '-', label="Ball {i} Energy".format(i=i+1), color=col, lw=1)
    #     # subplot.fill_between(t, bottom_stddev[i], top_stddev[i], color=col, alpha=0.25)
    

    # (energy_loss, fit_param)= find_amplitude_loss(t, total_energy, subplot, ball_mass)

    plt.title("Total Kinetic Energy Over Time")
    plt.plot(np.linspace(t[0],t[-1], num=len(total_energy)), total_energy, '-', label="Kinetic Energy", color='b', lw=1)
    plt.xlabel("$t$ [s]")
    plt.ylabel("Total Kinetic Energy [J]")
    plt.legend(ncols=len(x_positions)/2)
    plt.margins(x=0.01, y=0.01, tight=True)
    plt.xticks(np.arange(t[0], t[-1]+time_step, step=time_step))

    epsilon = 100
    # show_from = (index_at_release - epsilon) / fps
    # show_up_to = (index_at_release + epsilon) / fps

    # show_from = 2
    # show_up_to = 2.05
    
    # # t += index_at_release / fps 
    # subplots[2].set_title(f"Motion from ${show_from}\\,s \\to {show_up_to}\\,s$")
    # subplots[2].plot(t[int(show_from * fps):int(show_up_to*fps)], equilibrium_positions[0][int(show_from * fps):int(show_up_to*fps)], '-', label="Ball 1 Motion", color='g', lw=2)
    # subplots[2].plot(t[int(show_from * fps):int(show_up_to*fps)], equilibrium_positions[-1][int(show_from * fps):int(show_up_to*fps)], '-', label="Ball 5 Motion", color='c', lw=2)

    # # subplots[2].fill_between(t[int(show_from * fps):int(show_up_to*fps)], bottom_stddev[0][int(show_from * fps):int(show_up_to*fps)], top_stddev[0][int(show_from * fps):int(show_up_to*fps)], color=colour_from_index(0), alpha=0.25)
    # # subplots[2].fill_between(t[int(show_from * fps):int(show_up_to*fps)], bottom_stddev[-1][int(show_from * fps):int(show_up_to*fps)], top_stddev[-1][int(show_from * fps):int(show_up_to*fps)], color=colour_from_index(4), alpha=0.25)

    # # diff_0 = np.diff(energies[0])/np.diff(t[:-1])
    # # diff_4 = np.diff(energies[-1])/np.diff(t[:-1])

    # diff_0 = energies[0]
    # diff_4 = energies[-1]

    # # diff_0 = np.convolve(diff_0, np.ones(conv_width), 'valid') / conv_width
    # # diff_4 = np.convolve(diff_4, np.ones(conv_width), 'valid') / conv_width
    
    ## (diff_0_indices, diff_0_heights) = get_peak_indices(-diff_0)
    ## (diff_4_indices, diff_4_heights) = get_peak_indices(diff_4)

    ## subplots[1].plot(np.linspace(t[0],t[-1], num=len(diff_0)), diff_4, '-', label="Kinetic Energy", color='b', lw=1)
    ## for peak_index in diff_0_indices:
    ##     subplots[1].axvline((peak_index)/fps, color='r')

    ## for peak_index in diff_4_indices:
    ##     subplots[1].axvline((peak_index)/fps, color='g')

    ## min_length = min(len(diff_0_indices), len(diff_4_indices))
    ## consecutive_differences = np.abs(np.subtract(diff_0_indices[:min_length], diff_4_indices[:min_length]))

    # print(consecutive_differences)
    # print(np.where(    consecutive_differences < 55))

    # subplots[2].plot(t[int(show_from * fps):int(show_up_to*fps)], diff_0[int(show_from * fps):int(show_up_to*fps)], '-', label="Ball 1 Velocity", color='r', lw=2)
    # subplots[2].plot(t[int(show_from * fps):int(show_up_to*fps)], diff_4[int(show_from * fps):int(show_up_to*fps)], '-', label="Ball 5 Velocity", color='b', lw=2)
    # # length = min(len(diff_4), len(diff_0))
    # # t = np.linspace(t[0], t[-1], num=length) 
    # # subplots[2].plot(t[int(show_from * fps):int(show_up_to*fps)], np.subtract(diff_4,diff_0)[int(show_from * fps):int(show_up_to*fps)], '-', label="Ball -1 Motion", color='k', lw=2)
    # subplots[2].set_xlabel("$t$ [s]")
    # subplots[2].set_ylabel("$x$-Position [m]")
    # subplots[2].legend(ncols=len(x_positions)/2)
    # subplots[2].margins(x=0.01, y=0.01, tight=True)

    # limit = 0.005
    # subplots[2].set_ylim(-limit, limit)

    fig.tight_layout(pad=2.0)
    fig.savefig(f"{filename}-energies.pdf")
    plt.show()

plot_data(*parse_json())
