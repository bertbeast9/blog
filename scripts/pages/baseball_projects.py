import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
import matplotlib
import sys
import json
import datetime
import os
import pickle
from copy import copy
from scipy.optimize import linprog

# variables
dtype_dict = {"pitch_type":str,"game_date":str,"release_speed":"Float64",
                "release_pos_x":"Float64","release_pos_y":"Float64","release_pos_z":"Float64",
                "player_name":str,"batter":"Int64","pitcher":"Int64","events":str,"description":str,
                "spin_dir":"Float64","spin_rate_deprecated":"Float64","break_angle_deprecated":"Float64",
                "break_length_deprecated":"Float64","zone":"Int64","des":str,
                "game_type":str,"stand":str,"p_throws":str,"home_team":str,"away_team":str,
                "type":str,"hit_location":"Int64","bb_type":str,"balls":"Int64","strikes":"Int64",
                "game_year":"Int64","pfx_x":"Float64","pfx_z":"Float64","plate_x":"Float64","plate_z":"Float64",
                "on_3b":"Int64","on_2b":"Int64","on_1b":"Int64", "outs_when_up":"Int64",
                "inning":"Int64","inning_topbot":str,"hc_x":"Float64","hc_y":"Float64",
                "tfs_deprecated":"Int64","tfs_zulu_deprecated":"Int64",
                "umpire":"Int64","sv_id":"str","vx0":"Float64","vy0":"Float64","vz0":"Float64",
                "ax":"Float64","ay":"Float64","az":"Float64","sz_top":"Float64","sz_bot":"Float64",
                "hit_distance_sc":"Float64","launch_speed":"Float64","launch_angle":"Float64",
                "effective_speed":"Float64","release_spin_rate":"Float64","release_extension":"Float64",
                "game_pk":"Int64","pitcher":"Int64","fielder_2":"Int64","fielder_3":"Int64","fielder_4":"Int64",
                "fielder_5":"Int64","fielder_6":"Int64","fielder_7":"Int64","fielder_8":"Int64",
                "fielder_9":"Int64","estimated_ba_using_speedangle":"Float64",
                "estimated_woba_using_speedangle":"Float64","woba_value":"Float64","woba_denom":"Float64",
                "iso_value":"Float64","launch_speed_angle":"Float64","at_bat_number":"Int64",#"bapip_value":"Float64",
                "pitch_number":"Int64","pitch_name":str,"home_score":"Int64","away_score":"Int64","bat_score":"Int64",
                "fld_score":"Int64","post_home_score":"Int64","post_away_score":"Int64","post_bat_score":"Int64","post_fld_score":"Int64",
                "if_fielding_alignment":str,"of_fielding_alignment":str,"spin_axis":"Float64","delta_home_win_exp":"Float64",
                "delta_run_exp":"Float64","bat_speed":"Float64","swing_length":"Float64","estimated_slg_using_speedangle":"Float64",
                "delta_pitcher_run_exp":"Float64","hyper_speed":"Float64","home_score_diff":"Int64","bat_score_diff":"Int64",
                "home_win_exp":"Float64","bat_win_exp":"Float64","age_pit_legacy":"Int64","age_bat_legacy":"Int64","age_pit":"Int64","age_bat":"Int64",
                "n_thruorder_pitcher":"Int64","n_priorpa_thisgame_player_at_bat":"Int64","pitcher_days_since_prev_game":"Int64",
                "batter_days_since_prev_game":"Int64","pitcher_days_until_next_game":"Int64","batter_days_until_next_game":"Int64",
                "api_break_z_with_gravity":"Float64","api_break_x_arm":"Float64","api_break_x_batter_in":"Float64","arm_angle":"Float64",
                "attack_angle":"Float64","attack_direction":"Float64","swing_path_tilt":"Float64","intercept_ball_minus_batter_pos_x_inches":"Float64",
                "intercept_ball_minus_batter_pos_y_inches":"Float64"}
# variables

# functions
def add_strikezone(ax, bot_sz = 1.54, top_sz = 3.35):
    width = 17/12
    box_height = (1/3)*(top_sz-bot_sz)
    box_width = (1/3)*(width)
    zone7 = ax.add_patch(Rectangle((-width/2+0*box_width,bot_sz+0*box_height),box_width,box_height, ec="k", fc = "None"))
    zone8 = ax.add_patch(Rectangle((-width/2+1*box_width,bot_sz+0*box_height),box_width,box_height, ec="k", fc = "None"))
    zone9 = ax.add_patch(Rectangle((-width/2+2*box_width,bot_sz+0*box_height),box_width,box_height, ec="k", fc = "None"))
    zone4 = ax.add_patch(Rectangle((-width/2+0*box_width,bot_sz+1*box_height),box_width,box_height, ec="k", fc = "None"))
    zone5 = ax.add_patch(Rectangle((-width/2+1*box_width,bot_sz+1*box_height),box_width,box_height, ec="k", fc = "None"))
    zone6 = ax.add_patch(Rectangle((-width/2+2*box_width,bot_sz+1*box_height),box_width,box_height, ec="k", fc = "None"))
    zone1 = ax.add_patch(Rectangle((-width/2+0*box_width,bot_sz+2*box_height),box_width,box_height, ec="k", fc = "None"))
    zone2 = ax.add_patch(Rectangle((-width/2+1*box_width,bot_sz+2*box_height),box_width,box_height, ec="k", fc = "None"))
    zone3 = ax.add_patch(Rectangle((-width/2+2*box_width,bot_sz+2*box_height),box_width,box_height, ec="k", fc = "None"))
    zone13 = ax.add_patch(Polygon([(-width/2+0*box_width,bot_sz+1.5*box_height), (-width/2+0*box_width,bot_sz+0*box_height),(-width/2+1.5*box_width,bot_sz+0*box_height),(-width/2+1.5*box_width,bot_sz+-1*box_height),(-width/2+-1*box_width,bot_sz+-1*box_height),(-width/2+-1*box_width,bot_sz+1.5*box_height)], ec="k", fc = "None"))
    zone14 = ax.add_patch(Polygon([(-width/2+3*box_width,bot_sz+1.5*box_height),(-width/2+3*box_width,bot_sz+0*box_height),(-width/2+1.5*box_width,bot_sz+0*box_height),(-width/2+1.5*box_width,bot_sz+-1*box_height),(-width/2+4*box_width,bot_sz+-1*box_height),(-width/2+4*box_width,bot_sz+1.5*box_height)], ec="k", fc = "None"))
    zone11 = ax.add_patch(Polygon([(-width/2+0*box_width,bot_sz+1.5*box_height),(-width/2+0*box_width,bot_sz+3*box_height),(-width/2+1.5*box_width,bot_sz+3*box_height),(-width/2+1.5*box_width,bot_sz+4*box_height),(-width/2+-1*box_width,bot_sz+4*box_height),(-width/2+-1*box_width,bot_sz+1.5*box_height)], ec="k", fc = "None"))
    zone12 = ax.add_patch(Polygon([(-width/2+3*box_width,bot_sz+1.5*box_height),(-width/2+3*box_width,bot_sz+3*box_height),(-width/2+1.5*box_width,bot_sz+3*box_height),(-width/2+1.5*box_width,bot_sz+4*box_height),(-width/2+4*box_width,bot_sz+4*box_height),(-width/2+4*box_width,bot_sz+1.5*box_height)], ec="k", fc = "None"))
    strikezone = ax.add_patch(Polygon([(-width/2+0*box_width,bot_sz+0*box_height), (-width/2+3*box_width,bot_sz+0*box_height),(-width/2+3*box_width,bot_sz+3*box_height),(-width/2+0*box_width,bot_sz+3*box_height)], ec="r", fc = "None"))
    
    ax.set_xlim([-1.5*width,1.5*width])
    ax.set_ylim([0,5])
    zone = {1:zone1,2:zone2,3:zone3,4:zone4,5:zone5,6:zone6,7:zone7,8:zone8,9:zone9,11:zone11,12:zone12,13:zone13,14:zone14,"strikezone":strikezone}
    return ax, zone

def add_batter(ax,stand, off_plate = (17/12),batter_width = 1,batter_height = 6):
    if stand == "L":
        ax.add_patch(Rectangle([off_plate,0],batter_width, batter_height,ec = "g",fc = "g"))
    elif stand == "R":
        ax.add_patch(Rectangle([-off_plate-batter_width,0],batter_width, batter_height,ec = "g",fc = "g"))
    return ax
def add_pitcher(ax,p_throws):
    if p_throws == "L":
        ax.add_patch(Rectangle([0,4],0.5, 0.1,ec = "g",fc = "g"))
    elif p_throws == "R":
        ax.add_patch(Rectangle([-0.5,4],0.5, 0.1,ec = "g",fc = "g"))
    return ax
def visualize_policy(policy,split,state):
    global perc_ax, fig
    # for split in ["LL","LR","RL","RR"]:
    #     for state in policy[split].keys():
    fig = plt.figure()
    font = {
    'size'   : 6}

    matplotlib.rc('font', **font)
    # sz_ax = fig.add_subplot(121)
    # perc_ax = fig.add_subplot(122)
    # perc_ax.set_visible(False)
    sz_ax = fig.add_subplot(111)
    # perc_ax = fig.add_subplot(122)
    # perc_ax.set_visible(False)
    fig.canvas.flush_events()
    sz_ax, zone = add_strikezone(sz_ax)
    sz_ax = add_pitcher(sz_ax,split[0])
    sz_ax = add_batter(sz_ax,split[1])
    for zone_loc in list(range(1,10)) + list(range(11,15)):
        zone_usage = 0.0
        zone_pitches = []
        for item in policy[split][state]:
            control = item[0]
            percentage =  item[1]
            pitch, location = control.split(" ")
            if zone_loc == int(location):
                zone_usage += percentage
                zone_pitches.append([pitch,percentage])
        zone_usage = round(zone_usage,2)
        if zone_loc < 10:
            rx, ry = zone[int(zone_loc)].get_xy()
            cx = rx + zone[int(zone_loc)].get_width()/2.0
            cy = ry + zone[int(zone_loc)].get_height()/2.0
            string = f"Total: {zone_usage}%"
            for item in zone_pitches:
                string += f"\n {item[0]} {item[1]}%"
            sz_ax.annotate(string,(cx,cy),color='k',ha='center',va='center')
        elif zone_loc == 11 or zone_loc == 12:
            xys = zone[int(zone_loc)].get_xy()
            cx = xys[0,0] + 0.0
            cy = xys[0,1] + 2*zone[1].get_height()
            string = f"Total: {zone_usage}%"
            for item in zone_pitches:
                string += f"\n {item[0]} {item[1]}%"
            sz_ax.annotate(string,(cx,cy),color='k',ha='center',va='center')
        elif zone_loc == 13 or zone_loc == 14:
            xys = zone[int(zone_loc)].get_xy()
            cx = xys[0,0] + 0.0
            cy = xys[0,1] - 2*zone[1].get_height()
            string = f"Total: {zone_usage}%"
            for item in zone_pitches:
                string += f"\n {item[0]} {item[1]}%"
            sz_ax.annotate(string,(cx,cy),color='k',ha='center',va='center')
        

    def on_click(event,zone):
        global perc_ax, fig, split, state
        if event.button == 1:
            [zone[curr_zone].set_fc("none") for curr_zone in list(range(1,10)) + list(range(11,15))]
            for curr_zone in list(range(1,10)) + list(range(11,15)):
                if zone[curr_zone].contains_point([event.x,event.y]):
                    global perc_ax, fig
                    zone[curr_zone].set_fc("r")
                    # perc_ax.clear()
                    # perc_ax.set_visible(True)
                    zone_pitches = []
                    zone_percs = []
                    for item in policy[split][state]:
                        if int(item[0].split(" ")[1]) == curr_zone:
                            zone_pitches.append(item[0].split(" ")[0])
                            zone_percs.append(item[1])
                    # perc_ax.pie(zone_percs,labels = zone_pitches, colors = [pitch_color_dict[x] for x in zone_pitches],autopct='%1.1f%%')
                    # perc_ax.legend(loc = "best")
                    fig.canvas.draw()
                    fig.canvas.flush_events()

    cid = fig.canvas.mpl_connect("button_press_event", lambda event: on_click(event, zone))

    fig.suptitle(f"Split: {split} || State: {state}")
    # manager = plt.get_current_fig_manager()
    # manager.full_screen_toggle()
    # plt.savefig(f"C:/Users/sambe/Python/mlb/figures/{start_date.strftime("%Y_%m_%d")}_to_{end_date.strftime("%Y_%m_%d")}_{split}_{state}_{suffix}.jpg")
    # plt.close()
    # plt.show()

    return fig


def game_generator(startdate, enddate):
    # get folders
    day_folders = [f"C:/Users/sambe/Python/mlb/data/rawdata/games/{x}" for x in os.listdir("C:/Users/sambe/Python/mlb/data/rawdata/games") if os.path.isdir(f"C:/Users/sambe/Python/mlb/data/rawdata/games/{x}")]
    day_folders = sorted(day_folders)
    for folder in day_folders:
        folder_date = datetime.datetime.strptime(folder.split("/")[-1], "%Y-%m-%d")
        if folder_date < startdate or folder_date > enddate:
            continue
        for game_file in [f"{folder}/{x}" for x in os.listdir(folder)]:
            # print(game_file)
            df = pd.read_csv(game_file, dtype = dtype_dict)
            yield df
    return 


def get_state_action_transitions(start_date, end_date):
    trans_states = [f"({b},{s})" for b in range(4) for s in range(3)]
    term_states = ["out","walk","hit_by_pitch","single","double","triple","home_run"]
    all_states = trans_states + term_states
    card_S = len(all_states)
    # all_pitches = list(pitch_color_dict.keys())
    # all_pitches = ["FF","CH","SL"]
    all_pitches = ["FF","FS","FA","FO","CU","SL","FC","SI","ST","CH","KN","SV","KC","EP","PO"]
    all_locations = list(range(1,10)) + list(range(11,15))
    # all_locations = [2,5]
    all_splits = ["LL","LR","RL","RR"]
    trans_controls = [f"{pitch} {location}" for pitch in all_pitches for location in all_locations]
    # term_controls = ["O"]
    all_controls = trans_controls #+ term_controls
    card_U = len(all_controls)
    O = np.zeros([len(all_splits),card_U,card_S,card_S])
    allowed_trans = []
    for i in range(card_S):
        if i >= len(trans_states): # absorbing states
            O[:,:,i,i] = 1
        else: # transient states
            curr_state = all_states[i]
            for j in range(card_S):
                next_state = all_states[j]
                if curr_state == next_state and curr_state[3] == "2": # foul ball with two strikes
                    O[:,:,i,j] = 1
                    allowed_trans.append(f"{curr_state} -> {next_state}")
                elif curr_state != next_state:
                    if next_state in term_states:
                        if curr_state[1] == "3":
                            O[:,:,i,j] = 1
                            allowed_trans.append(f"{curr_state} -> {next_state}")
                        elif next_state != "walk":
                            O[:,:,i,j] = 1
                            allowed_trans.append(f"{curr_state} -> {next_state}")
                    else:
                        if int(next_state[1]) == 1 + int(curr_state[1]) and next_state[3] == curr_state[3]: # ball thrown
                            O[:,:,i,j] = 1
                            allowed_trans.append(f"{curr_state} -> {next_state}")
                        elif int(next_state[3]) == 1 + int(curr_state[3]) and next_state[1] == curr_state[1]: # strike thrown
                            O[:,:,i,j] = 1
                            allowed_trans.append(f"{curr_state} -> {next_state}")
    # print(O)
    # print(O.shape)
    # input()
    G = np.zeros([len(all_splits),card_U,card_S])
    N = np.ones([len(all_splits),card_U,card_S])
    for l in range(len(all_splits)):
        for i in range(card_S):
            curr_state = all_states[i]
            if i < len(trans_states):
                for j in range(card_U):
                    if all_pitches[j//len(all_locations)] == "FF":
                        G[l,j,i] = 0.01
                        N[l,j,i] = 1
                    else:
                        G[l,j,i] = 0.01
                        N[l,j,i] = 1
                    # G[1,i] = 0.01
                    # G[2,i] = 0.01
    # # print(G)
    # # print(G.shape)
    # # input()
    
    event_2_state_dict = {"strikeout":"out","home_run":"home_run","foul_popout":"out","field_out":"out","single":"single","double":"double","triple":"triple","walk":"walk","hit_by_pitch":"hit_by_pitch","force_out":"out","grounded_into_double_play":"out","sac_bunt":"out","sac_fly":"out","field_error":"out","double_play":"out","triple_play":"out","fielders_choice":"out","triple_play":"out","sac_fly_double_play":"out","fielders_choice_out":"out","sac_bunt_double_play":"out","ejection":"out","truncated_pa":"out","catcher_interf":"out","strikeout_double_play":"out","intent_walk":"walk","game_advisory":"out"}
    event_2_bases_dict = {"out": -1, "walk": 0, "hit_by_pitch": 0, "single": 0, "double": 0, "triple": 0, "home_run": 0}
    filename = f"./mdls/{start_date.strftime("%Y_%m_%d")}_to_{end_date.strftime("%Y_%m_%d")}_state_trans_probs.pkl"
    if os.path.isfile(filename):
        (P, G, all_states, all_controls, all_locations, all_pitches, all_splits) = pickle.load(open(filename,"rb"))
    else:
        for game in game_generator(start_date, end_date):
            print(game.iloc[0,:].game_date)
            # map states
            game["game_state"] = game.apply(lambda x: f"({x.balls},{x.strikes})"  ,axis = 1)
            game["events"] = game["events"].apply(lambda x: x if pd.isna(x) else event_2_state_dict[x])
            for idx, row in game.iterrows():
                curr_state = row.game_state
                # check if there is a pitch_type defined
                if not pd.isna(row.pitch_type) and not pd.isna(row.zone):
                    curr_control = f"{row.pitch_type} {row.zone}"
                    if pd.isna(row.events) and idx+1 < game.shape[0]:
                        next_state = game.iloc[idx+1].game_state
                        if curr_control in all_controls and curr_state in all_states and next_state in all_states:
                            i = all_controls.index(curr_control)
                            j = all_states.index(curr_state)
                            k = all_states.index(next_state)
                            l = all_splits.index(f"{row.p_throws}{row.stand}")
                            if O[l,i,j,k] > 0:
                                O[l,i,j,k] += 1

                            else:
                                print("not possible")
                                print(row.events)
                                print(f"{curr_control} {curr_state}, {next_state}")
                                print(f"{i}, {j}, {k}")
                            #     input()
                    elif not pd.isna(row.events):
                        next_state = row.events
                        if curr_control in all_controls and curr_state in all_states and next_state in all_states:
                            i = all_controls.index(curr_control)
                            j = all_states.index(curr_state)
                            k = all_states.index(next_state)
                            l = all_splits.index(f"{row.p_throws}{row.stand}")
                            cost = event_2_bases_dict[next_state]  + row.post_bat_score - row.bat_score
                            if O[l,i,j,k] > 0:
                                O[l,i,j,k] += 1
                                G[l,:,k] += cost
                                N[l,:,k] += 1
                                # print(f"{curr_state} | {next_state} | {G[l,:,k]}| {cost}")
                                # input()
                                
                            else:
                                print("not possible")
                                print(row.events)
                                print(f"{curr_control} {curr_state}, {next_state}")
                                print(f"{i}, {j}, {k}")
                                # input()
        P = copy(O)
        for l in range(len(all_splits)):
            for i in range(len(trans_controls)):
                for j in range(card_S):
                    row_sum = np.sum(P[l,i,j,:])
                    P[l,i,j,:] /= row_sum
        G = G/N
        for l in range(len(all_splits)):
            print(f"{all_splits[l]}")
            for k in range(card_S):
                state = all_states[k]
                if state in term_states:
                    print(f"{state} | avg. cost: {G[l,0,k]}")
        # print(G)
        # input()
            # for j in range(card_S):
            #     next_state = all_states[j]
            #     if next_state in term_states:
            #         row_sum = 0
            #         for i in range(card_U):
            #             row_sum += np.sum(O[l,i,:,j])
            #         G[l,:,j] /= row_sum
            #         print(f"{next_state} | cost: {G[l,:,j]}")
        pickle.dump((P, G, all_states, all_controls, all_locations, all_pitches, all_splits),open(filename,"wb"))
    
        
    return P, G, all_states, all_controls, all_locations, all_pitches, all_splits


def pitch_solver_main_count_only(pitch_constraints,location_constraints):
    start_date = datetime.datetime(2008,3,1)
    # start_date = datetime.datetime(2025,6,1)
    # end_date = datetime.datetime(2025,7,1)
    end_date = datetime.datetime(2025,11,1)
    # # # # opt_policy = json.load(open(f"C:/Users/sambe/Python/mlb/data/mdls/{start_date.strftime("%Y_%m_%d")}_to_{end_date.strftime("%Y_%m_%d")}_opt_policy.json","r"))
    # # # # visualize_policy(opt_policy,start_date,end_date,suffix = "")
    print(sys.path)
    folder = "./mdls/count_only_opt_policies"
    # filename = os.path.join(folder,"CNN_VM_CMP.jpg")
    # st.image(filename)
    found_filename = ""
    i = 0
    st.write(sys.path)
    for i,filename in enumerate(os.listdir(folder)):
        filename = os.path.join(folder,filename)
        opt_policy = json.load(open(filename,"r"))
        # print(opt_policy)
        if opt_policy["constraints"]["pitch_perc_ub_total"] == pitch_constraints[0] and \
            opt_policy["constraints"]["pitch_perc_ub_state"] == pitch_constraints[1] and \
            opt_policy["constraints"]["location_perc_ub_total"] == location_constraints[0] and \
            opt_policy["constraints"]["location_perc_ub_state"] == location_constraints[1]:
            found_filename = filename
            break
    if len(found_filename) < 10:
        filename = os.path.join(folder, f"{start_date.strftime("%Y_%m_%d")}_to_{end_date.strftime("%Y_%m_%d")}_opt_policy_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}")
        print(f"creating new constraints file {filename}")
        # print(f"C:/Users/sambe/Python/mlb/data/mdls/{start_date.strftime("%Y_%m_%d")}_to_{end_date.strftime("%Y_%m_%d")}_opt_policy.json")
        # input()
        P, G, all_states, all_controls, all_locations, all_pitches, all_splits = get_state_action_transitions(start_date, end_date)
        opt_policy = dict()
        opt_policy["constraints"] = dict()
        opt_policy["constraints"]["pitch_perc_ub_total"] = pitch_constraints[0]
        opt_policy["constraints"]["pitch_perc_ub_state"] = pitch_constraints[1]
        opt_policy["constraints"]["location_perc_ub_total"] = location_constraints[0]
        opt_policy["constraints"]["location_perc_ub_state"] = location_constraints[1]
        for l, split in enumerate(all_splits):
            print(f"Optimal policy for {split[0]}HP vs {split[1]}HH")
            opt_policy[split] = dict()
            # print(G)
            card_S = len(all_states)
            card_U = len(all_controls)
            card_vars = len(all_states)*len(all_controls)
            c = np.append(G[l,:,:].T.reshape(-1,),np.zeros((card_vars,)),axis = 0)
            var_bounds = [(0,None) for i in range(card_vars*2)]
            
            A_eq1 = np.zeros([card_S,card_vars])
            b_eq1 = np.zeros([card_S])
            for j in range(card_S):
                state = all_states[j]
                add_ind = np.arange(j*card_U,(j+1)*card_U)
                A_eq1[j,add_ind] += 1
                for i in range(card_S):
                    prev_state = all_states[i]
                    sub_ind = np.arange(i*card_U,(i+1)*card_U)
                    A_eq1[j,sub_ind] -= P[l,:,i,j]
            A_eq1 = np.concat([A_eq1,np.zeros([card_S,card_vars])],axis = 1)
            A_eq2a = np.zeros([card_S,card_vars])
            for j in range(card_S):
                state = all_states[j]
                add_ind = np.arange(j*card_U,(j+1)*card_U)
                A_eq2a[j,add_ind] += 1
            A_eq2b = np.zeros([card_S,card_vars])
            b_eq2 = np.ones(shape = (card_S,))#np.random.uniform(size=(card_S,))
            b_eq2 /= np.sum(b_eq2)
            for j in range(card_S):
                state = all_states[j]
                add_ind = np.arange(j*card_U,(j+1)*card_U)
                A_eq2b[j,add_ind] += 1
                for i in range(card_S):
                    prev_state = all_states[i]
                    sub_ind = np.arange(i*card_U,(i+1)*card_U)
                    A_eq2b[j,sub_ind] -= P[l,:,i,j]
            A_eq2 = np.concat([A_eq2a,A_eq2b],axis = 1)


            # # # # # # # # # # # percent pitch usage
            A_ub0 = np.zeros([len(all_pitches),card_vars*2])
            b_ub0 = np.zeros([len(all_pitches)])
            for i, pitch in enumerate(all_pitches):
                A_ub0[i,card_vars:-7*card_U] = -pitch_constraints[0][all_pitches[i]]
                for j in range(i*(len(all_locations)), card_vars-7*card_U, len(all_locations)*len(all_pitches)):
                    A_ub0[i,j+card_vars:j+card_vars+len(all_locations)] += 1

            A_ub1 = np.zeros([len(all_pitches)*len(all_states),card_vars])
            b_ub1 = np.zeros([len(all_pitches)*len(all_states)])
            for i in range(len(all_pitches)*card_S):
                idx = i // card_S
                jdx = i % len(all_pitches)
                A_ub1[i,(i // (len(all_pitches)))*len(all_pitches)*len(all_locations):(i // (len(all_pitches)))*len(all_pitches)*len(all_locations) + len(all_pitches)*len(all_locations)] = -pitch_constraints[1][all_pitches[jdx]]
                A_ub1[i,i*len(all_locations):(i+1)*len(all_locations)] += 1
            A_ub1 = np.concat([np.zeros([len(all_pitches)*len(all_states),card_vars]),A_ub1],axis = 1)
            # # # # # # # # # # # percent location usage
            A_ub2 = np.zeros([len(all_locations),card_vars*2])
            b_ub2 = np.zeros([len(all_locations)])
            for i, location in enumerate(all_locations):
                A_ub2[i,card_vars:-7*card_U] = -location_constraints[0][f"{location}"]
                for j in range(i, card_vars-7*card_U, len(all_locations)):
                    A_ub2[i,j+card_vars] += 1

            A_ub3 = np.zeros([len(all_locations)*len(all_states),card_vars])
            b_ub3 = np.zeros([len(all_locations)*len(all_states)])
            for i in range(len(all_locations)*card_S):
                idx = i // len(all_locations)
                jdx = i % len(all_locations)
                add_ind = np.arange(jdx,len(all_pitches)*len(all_locations),len(all_locations)) + idx * card_U
                sub_ind = np.arange(idx * card_U,(idx+1) * card_U)            
                A_ub3[i,sub_ind] = -location_constraints[1][f"{all_locations[jdx]}"]
                A_ub3[i,add_ind] += 1
            A_ub3 = np.concat([np.zeros([len(all_locations)*len(all_states),card_vars]),A_ub3],axis = 1)
            

            A_eq = np.concat([A_eq1,A_eq2],axis = 0)
            b_eq = np.concat([b_eq1,b_eq2],axis = 0)
            A_ub = np.concat([A_ub0,A_ub1,A_ub2,A_ub3],axis = 0)
            b_ub = np.concat([b_ub0,b_ub1,b_ub2,b_ub3],axis = 0)
            res = linprog(c,A_ub = A_ub, b_ub = b_ub,A_eq = A_eq,b_eq = b_eq, bounds = var_bounds)#
            qs = res.x[:card_vars]
            rs = res.x[card_vars:]
            for i in range(card_vars):
                idx = i % card_U
                control = all_controls[idx]
                jdx = i // card_U
                state = all_states[jdx]
                if qs[i] > 0 and rs[i] > 0:
                    print(f"state {state} | control {control} | q: {qs[i]} | r: {rs[i]}")
                    input()
            # print(res)
            # print(res.fun)
            start_controls = 0
            end_controls = card_U
            for i in range(card_vars):
                control_idx = i % card_U
                state_idx = i // card_U
                state_usage = 100*rs[i]/np.sum(rs[start_controls:end_controls])
                if state_usage > 0 and "(" in all_states[state_idx]:
                    if all_states[state_idx] in opt_policy[split].keys():
                        opt_policy[split][all_states[state_idx]].append((all_controls[control_idx],round(state_usage,2)))
                    else:
                        opt_policy[split][all_states[state_idx]] = [(all_controls[control_idx],round(state_usage,2))]
                if (i+1) % card_U == 0:
                    start_controls += card_U
                    end_controls += card_U
            # for state in opt_policy[split].keys():
            #     print(f"{state}")
            #     for item in opt_policy[split][state]:
            #         print(f"\t{item[0]} | {item[1]}%")
            # for pitch in all_pitches:
            #     total_usage = 0.0
            #     for control in [f"{pitch} {location}" for location in all_locations]:
            #         control_idx = all_controls.index(control)
            #         for i in range(card_vars-7*card_U):
            #             control_jdx = i % card_U
            #             if control_jdx == control_idx:
            #                 total_usage += 100*rs[i]/np.sum(rs[:-7*card_U])
            #     total_usage = round(total_usage,2)
            #     if total_usage > 0:
            #         print(f"Pitch {pitch} | used {total_usage}% of the time")
            # for location in all_locations:
            #     total_usage = 0.0
            #     for control in [f"{pitch} {location}" for pitch in all_pitches]:
            #         control_idx = all_controls.index(control)
            #         for i in range(card_vars-7*card_U):
            #             control_jdx = i % card_U
            #             if control_jdx == control_idx:
            #                 total_usage += 100*rs[i]/np.sum(rs[:-7*card_U])
            #     total_usage = round(total_usage,2)
            #     if total_usage > 0:
            #         print(f"location {location} | used {total_usage}% of the time")
            # print(f"Optimal policy for {split[0]}HP vs {split[1]}HH | final func. value: {res.fun}\n\n")
        json.dump(opt_policy,open(filename,"w"),indent = 4)
        # visualize_policy(opt_policy,start_date,end_date,suffix = "pitch_location_constrained")
    else:
        opt_policy = json.load(open(found_filename,"r"))
        # print(opt_policy)
    # visualize_policy(opt_policy,start_date,end_date,suffix = "")
    return opt_policy

# @st.cache_data
def simulate_next_pitch_count_only(split,balls,strikes,n_sims,pitch_constraints,location_constraints):
    # solve with constraints
    # print("hey")
    count = f"({balls},{strikes})"
    opt_policy = pitch_solver_main_count_only(pitch_constraints,location_constraints)
    fig = visualize_policy(opt_policy,split,count)
    st.pyplot(fig)
    # print(opt_policy[split][count])
    # opt_controls = [x[0] for x in opt_policy[split][count]]
    # opt_probs = np.array([x[1] for x in opt_policy[split][count]])
    # cumsum_probs = np.cumsum(opt_probs)
    # rand_nums = 100*np.random.uniform(size = (n_sims,))
    # samples = []
    # for i in range(n_sims):
    #     samples.append(opt_controls[np.argwhere(cumsum_probs >= rand_nums[i])[0][0]])
    # print(cumsum_probs)
    # print(rand_nums)
    # print(samples)

    return 
# functions

st.header("Baseball Projects",divider=True)
st.sidebar.markdown("# Baseball Projects")
st.sidebar.subheader("Pitch Solver (Count Only)")
st.sidebar.subheader("Pitch Solver (Expanded)")
st.subheader("Pitch Solver (Count Only)")
st.write("Okay, let's start with the problem at hand. A begin decision before each " \
"pitch is what pitch to throw. We aim to model this as a Markov Decision Process (MDP) " \
"with the states being the state of the game. As a starting point, we are only going " \
"to consider the count as the state of the at-bat. Future work will expand the state-space " \
"to include the runners and outs. The full state space consists of the current at-bat's " \
"count, termed *transient states*, and the results of these at-bats, termed *absorbing* " \
"states. Again, for simplicity, we break the results of each at-bat into one of seven " \
"outcomes (out, walk, hit by pitch, single, double, triple, home run). Later, these " \
"outcomes will be further expanded to include sac flys, double plays, etc. The full " \
"state-space of this simple problem is the union of the *transient* and *absorbing* " \
"state-spaces.")
st.latex(r'''
    x_{k} \in S = \{(b,s) | b \in \{0,1,2,3 \}, s \in \{0,1,2 \} \} \cup \{\text{out,walk,hit by pitch, single, double, triple, home run} \}
    ''')
st.latex(r'''
    x_{k} \in S = S_{t} \cup S_{a}         
''')
st.latex(r'''
    |S| = 12 + 7 = 19         
''')
st.write("The control at our disposal is just the pitch and location that we call to " \
"be thrown by the pitcher. Another assumption made by this model formulation is that " \
"the pitcher has perfect command: if a 4-Seam Fastball (FF) is called in the 7 zone, " \
"then that is exactly the pitch that is thrown. Later, we aim to incorporate more " \
"uncertainty in the pitch location (i.e. model a pitcher with less command of their " \
"pitches). The pitch locations are defined as zones according to " \
"[Baseball Savant](%s) and is depicted in Figure 1." % "https://baseballsavant.mlb.com/")
st.image("./figures/Savant_Strikezone.png",width='content',caption="Zone definitions according Baseball Savant")
st.write("Every pitcher has a repertoire of pitches at their disposal. " \
"We will assume for this example that our pitcher has three pitches that " \
"they command well: a 4-Seam Fastball (FF), a Changeup (CH), and a Sweeper " \
"(ST). The control-space is then designated below.")
st.latex(r'''
    u_{k} \in U = \{(p,l) | p \in \{ \text{FF},\text{CH},\text{ST}\}, l \in \{1,2,3,4,5,6,7,8,9,11,12,13,14\}\}
''')
st.latex(r'''
|U| = 3 \times 13 = 39
''')
st.write("Of course, the control actions of the pitcher once they have " \
"reached an *absorbing* state has no effect on the outcome that had " \
"occurred, and this will be represented in the one-step cost seen later " \
"in this document.\n\nThe dynamic model is defined by")
st.latex(r'''
    x_{0} = (0,0) \\
    x_{k+1} = f(x_{k},u_{k},w_{k}) = P(x_{k+1}|x_{k},u_{k},w_{k}) \\
    P(x_{k+1}|x_{k},u_{k},w_{k}) = \frac{\text{number of transitions from \(x_{k}\) to \(x_{k+1}\) given \(u_{k}\) was called}}{\text{number of occurrences in which game was in state \(x_{k}\) and \(u_{k}\) was called}}  \ \ \ \ \forall x_{k} \in S_t, u_{k} \in U(x_{k})\\
    P(x_{k}|x_{k},u_{k},w_{k}) = 1 \ \ \ \ \forall x_{k} \in S_a, u_{k} \in U(x_{k})\\
''')
st.write("The one-step cost is defined by")
st.latex(r'''
    g(x_{k},u_k) = 0.01 \ \ \ \ \forall x_k \in S_t, u_{k} \in U \\
    g(x_{k},u_k) = C_{x_{k}} \ \ \ \ \forall x_k \in S_a, u_{k} \in U
''')
st.write("Now, every at bat is going to start in the state, $(0,0)$ " \
"and progress into one of the 7 absorbing states. Each of these absorbing " \
"will have an associated cost or penalty, $C_{x_{k}}$ for reaching that " \
"state proportional to that state's effect on the fielding team's " \
"likelihood of winning. With our dataset, we utilized calculated the " \
"average runs scored after each of these events occurred. Furthermore, " \
"we encouraged the optimal policy to seek more 'out' events by setting " \
"$C_{x_{k}} = -1,x_{k}=out$. This MDP is clearly not " \
"*unichain*, but we can rely on the Linear Programming (LP) " \
"solver to dictate the optimal policy for this situation.\n\nThe " \
"dual LP formulation is")
st.latex(r'''
    min \sum_{x_{k} \in S} \sum_{u_{k} \in U(x_{k})} g(x_{k},u_{k})q(x_{k},u_{k}) \\
    s.t. \ \ \ q(x_{k},u_{k}),r(x_{k},u_{k}) \geq 0 \ \ \ \ \forall x_{k} \in S, u_{k} \in U(x_{k}) \\
    \sum_{u_{j} \in U(x_{j})} q(x_{j},u_{j}) = \sum_{x_{k} \in S} \sum_{u_{k} \in U(x_{k})} q(x_{k},u_{k}) P(x_{j}|x_{k},u_{k},w_{k}) \ \ \ \ \forall x_{j} \in S \\
    \sum_{u_{j} \in U(x_{j})} q(x_{j},u_{j}) + \sum_{u_{j} \in U(x_{j})} r(x_{j},u_{j}) = \beta_{x_{j}} + \sum_{x_{k} \in S} \sum_{u_{k} \in U(x_{k})} r(x_{k},u_{k}) P(x_{j}|x_{k},u_{k},w_{k}) \ \ \ \ \forall x_{j} \in S \\
''')
st.write("So, $q(x_{k},u_{k})$ can be considered to be the long-run " \
"'state-action' frequency for the states that are recurrent states and " \
"$r(x_{k},u_{k})$ can be considered to be the long-run 'state-action' " \
r"frequency for the states that are transient states. $ \beta_{x_{j}}$ " \
"is any positive scalar value such that all " \
r"$\sum_{x_j \in S} \beta_{x_{j}} = 1 $. Each state-action is either " \
"recurrent or transient, and for this problem, we only care about the " \
"policy for the transient states since the recurrent states (out, " \
"walk, etc.) have no more decisions to be made. \n\nThe result of the LP " \
"will provide this optimal policy via")
st.latex(r'''
    P^*(u_{k}|x_{k}) = \frac{r^*(x_{k},u_{k})}{\sum_{u_{j} \in U(x_{k})} r^*(x_{k},u_{j})}
''')
st.write("Now, for a better breakdown of optimal pitch selection, it " \
"is best to divide the at-bats by their splits (a left-handed pitcher " \
"versus a right-handed hitter [LHP v RHH], etc.) since certain pitches are " \
"more effective against certain handed batters. For example, a 'back-foot' " \
"slider is very effective for a matchup with opposite handed batters and " \
"pitchers (LHP v RHH or RHP vs LHH) since the slider will appear to be a " \
"strike out of the hand and break towards the batter's backfoot (and thus " \
"become almost unhittable). The results will be broken down by at-bat " \
"splits.\n\nAn advantage of this problem formulation is the ability to " \
"constrain the solution. This means that the modeler can add constraints " \
"to the optimal policy such that a pitcher does not use a particular pitch " \
"over $p_{ub,total} = 50\\%$ of the time and/or a pitcher does not use a " \
"particular pitch in a particular count over $p_{ub,x_{k}} = 50\\%$ of " \
"the time. The added constraint looks like")
st.latex(r'''
    \sum_{x_{k} \in S} \sum_{u_{k} \in U(x_{k})} I_{u_{k}}(p)r(x_{k},u_{k}) \leq p_{ub,total}\sum_{x_{k} \in S} \sum_{u_{k} \in U(x_{k})} r(x_{k},u_{k}) \ \ \ \ \forall p \in \{\text{FF},\text{CH},\text{ST}\} \\
    \sum_{u_{k} \in U(x_{k})} I_{u_{k}}(p)r(x_{k},u_{k}) \leq p_{ub,x_{k}} \sum_{u_{k} \in U(x_{k})} r(x_{k},u_{k}) \ \ \ \ \forall p \in \{\text{FF},\text{CH},\text{ST}\}, x_{k} \in S \\         
''')
st.write("Working with historical data from the 2008 to 2025 seasons, " \
"we were able to extract out the state-action transition probabilities " \
"$P(x_{k+1}|x_{k},u_{k},w_{k})$, and the average cost of each absorbing " \
"state (event) $C_{x_{k}}$. With no constraints on the percentage of " \
"time a pitch is thrown and assuming this pitcher has access to every " \
"known pitch (see Table below), the results of this optimal policy can " \
"be simulated below.")
pitch_dict = {"FF":"4-Seam Fastball","SI":"Sinker (2-Seam)","FC":"Cutter",
          "CH":"Changeup","FS":"Split-finger","FO":"Forkball","SC":"Screwball",
          "CU":"Curveball","KC":"Knuckle Curve","CS":"Slow Curve","SL":"Slider",
          "ST":"Sweeper","SV":"Slurve","KN":"Knuckleball","EP":"Eephus","FA":"Other",
          "IN":"Intentional Ball","PO":"Pitchout"}
st.table(pitch_dict)
st.divider()
col1, col2, col3, col4 = st.columns([1,1,1,1])
with col1:
    split = st.selectbox("Split",["LHP vs LHH","LHP vs RHH","RHP vs LHH","RHP vs RHH"])
with col2:
    n_pitch_sims = st.number_input("Number of sim. pitches",min_value=1, max_value=1000)
with col3:
    n_balls = st.number_input("Number of balls",min_value=0, max_value=3)
with col4:
    n_strikes = st.number_input("Number of strikes",min_value=0, max_value=2)
st.divider()
split_dict = {"LHP vs LHH":"LL","LHP vs RHH":"LR","RHP vs LHH":"RL","RHP vs RHH":"RR"}
split = split_dict[split]

FF_perc = st.slider("FF % usage",min_value = 0, max_value = 100, value = 0)
SI_perc = st.slider("SI % usage",min_value = 0, max_value = 100, value = 0)
FC_perc = st.slider("FC % usage",min_value = 0, max_value = 100, value = 0)
CH_perc = st.slider("CH % usage",min_value = 0, max_value = 100, value = 0)
FS_perc = st.slider("FS % usage",min_value = 0, max_value = 100, value = 0)
FO_perc = st.slider("FO % usage",min_value = 0, max_value = 100, value = 0)
SC_perc = st.slider("SC % usage",min_value = 0, max_value = 100, value = 0)
CU_perc = st.slider("CU % usage",min_value = 0, max_value = 100, value = 0)
KC_perc = st.slider("KC % usage",min_value = 0, max_value = 100, value = 0)
CS_perc = st.slider("CS % usage",min_value = 0, max_value = 100, value = 0)
SL_perc = st.slider("SL % usage",min_value = 0, max_value = 100, value = 0)
ST_perc = st.slider("ST % usage",min_value = 0, max_value = 100, value = 0)
SV_perc = st.slider("SV % usage",min_value = 0, max_value = 100, value = 0)
KN_perc = st.slider("KN % usage",min_value = 0, max_value = 100, value = 0)
EP_perc = st.slider("EP % usage",min_value = 0, max_value = 100, value = 0)
FA_perc = st.slider("FA % usage",min_value = 0, max_value = 100, value = 0)
IN_perc = st.slider("IN % usage",min_value = 0, max_value = 100, value = 0)
PO_perc = st.slider("PO % usage",min_value = 0, max_value = 100, value = 0)
pitch_sum = FF_perc + SI_perc + FC_perc + CH_perc + FS_perc + FO_perc + SC_perc + CU_perc + KC_perc + CS_perc + SL_perc + ST_perc + SV_perc + KN_perc + EP_perc + FA_perc + IN_perc + PO_perc 
pitch_sum_gt_100 = pitch_sum >= 100
if not pitch_sum_gt_100:
    st.warning(f"The sum of the pitch percentages needs to be greater than or equal to 100%")

st.divider()
col1, col2 = st.columns([1,1])
with st.container():
    loc11_perc = col1.slider("loc 11 % usage",min_value = 0, max_value = 100, value = 100)
    loc12_perc = col2.slider("loc 12 % usage",min_value = 0, max_value = 100, value = 100)
col3, col4, col5 = st.columns([1,1,1])
with st.container():
    loc1_perc = col3.slider("loc 1 % usage",min_value = 0, max_value = 100, value = 100)
    loc2_perc = col4.slider("loc 2 % usage",min_value = 0, max_value = 100, value = 100)
    loc3_perc = col5.slider("loc 3 % usage",min_value = 0, max_value = 100, value = 100)
col6, col7, col8 = st.columns([1,1,1])
with st.container():
    loc4_perc = col6.slider("loc 4 % usage",min_value = 0, max_value = 100, value = 100)
    loc5_perc = col7.slider("loc 5 % usage",min_value = 0, max_value = 100, value = 100)
    loc6_perc = col8.slider("loc 6 % usage",min_value = 0, max_value = 100, value = 100)
col9, col10, col11 = st.columns([1,1,1])
with st.container():
    loc7_perc = col9.slider("loc 7 % usage",min_value = 0, max_value = 100, value = 100)
    loc8_perc = col10.slider("loc 8 % usage",min_value = 0, max_value = 100, value = 100)
    loc9_perc = col11.slider("loc 9 % usage",min_value = 0, max_value = 100, value = 100)
col12, col13 = st.columns([1,1])
with st.container():
    loc13_perc = col12.slider("loc 13 % usage",min_value = 0, max_value = 100, value = 100)
    loc14_perc = col13.slider("loc 14 % usage",min_value = 0, max_value = 100, value = 100)


# double check that percentages are >= 100%
loc_sum = loc1_perc + loc2_perc + loc3_perc + loc4_perc + loc5_perc + loc6_perc + loc7_perc + loc8_perc + loc9_perc + loc11_perc + loc12_perc + loc13_perc + loc14_perc 
loc_sum_gt_100 = loc_sum >= 100
if not loc_sum_gt_100:
    st.warning(f"The sum of the location percentages needs to be greater than or equal to 100%")

# pitch_constraints = {"FF":}
pitch_percentage_ub_total = {"FF":FF_perc/100,"FS":FS_perc/100,"FA":FA_perc/100,"FO":FO_perc/100,"CU":CU_perc/100,"SL":SL_perc/100,"FC":FC_perc/100,"SI":SI_perc/100,"ST":ST_perc/100,"CH":CH_perc/100,"KN":KN_perc/100,"SV":SV_perc/100,"KC":KC_perc/100,"EP":EP_perc/100,"PO":PO_perc/100}
pitch_percentage_ub_state = {"FF":FF_perc/100,"FS":FS_perc/100,"FA":FA_perc/100,"FO":FO_perc/100,"CU":CU_perc/100,"SL":SL_perc/100,"FC":FC_perc/100,"SI":SI_perc/100,"ST":ST_perc/100,"CH":CH_perc/100,"KN":KN_perc/100,"SV":SV_perc/100,"KC":KC_perc/100,"EP":EP_perc/100,"PO":PO_perc/100}
location_percentage_ub_total = {"1":loc1_perc/100,"2":loc2_perc/100,"3":loc3_perc/100,"4":loc4_perc/100,"5":loc5_perc/100,"6":loc6_perc/100,"7":loc7_perc/100,"8":loc8_perc/100,"9":loc9_perc/100,"11":loc11_perc/100,"12":loc12_perc/100,"13":loc13_perc/100,"14":loc14_perc/100}
location_percentage_ub_state = {"1":loc1_perc/100,"2":loc2_perc/100,"3":loc3_perc/100,"4":loc4_perc/100,"5":loc5_perc/100,"6":loc6_perc/100,"7":loc7_perc/100,"8":loc8_perc/100,"9":loc9_perc/100,"11":loc11_perc/100,"12":loc12_perc/100,"13":loc13_perc/100,"14":loc14_perc/100}


if st.button("Simulate Pitch"):
    st.write(f"Simulating next pitch {n_pitch_sims} time(s)")
    simulate_next_pitch_count_only(split,n_balls,n_strikes,n_pitch_sims,pitch_constraints=(pitch_percentage_ub_total,pitch_percentage_ub_state),location_constraints=(location_percentage_ub_total,location_percentage_ub_state))