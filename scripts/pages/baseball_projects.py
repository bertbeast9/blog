import streamlit as st
import numpy as np
import sys
sys.path.append("C:/Users/sambe/Python")
from mlb.py_bet_mlb.simulate.update_models import pitch_solver_main_count_only, visualize_policy
# functions
# @st.cache_data
def simulate_next_pitch_count_only(split,balls,strikes,n_sims,pitch_constraints,location_constraints):
    # solve with constraints
    # print("hey")
    count = f"({balls},{strikes})"
    opt_policy = pitch_solver_main_count_only(pitch_constraints,location_constraints)
    # fig = visualize_policy(opt_policy,split,count)
    # st.pyplot(fig)
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