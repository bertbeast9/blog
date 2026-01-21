import streamlit as st

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
st.image("./figures/Savant_Strikezone.png",use_container_width=True,caption="Zone definitions according Baseball Savant")
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
"over $p_{ub,total} = 50\%$ of the time and/or a pitcher does not use a " \
"particular pitch in a particular count over $p_{ub,x_{k}} = 50\%$ of " \
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
st.table({"FF":"4-Seam Fastball","SI":"Sinker (2-Seam)","FC":"Cutter",
          "CH":"Changeup","FS":"Split-finger","FO":"Forkball","SC":"Screwball",
          "CU":"Curveball","KC":"Knuckle Curve","CS":"Slow Curve","SL":"Slider",
          "ST":"Sweeper","SV":"Slurve","KN":"Knuckleball","EP":"Eephus","FA":"Other",
          "IN":"Intentional Ball","PO":"Pitchout"})
st.divider()
col1, col2 = st.columns([1,1])
with col1:
    if st.button("Reset At-Bat"):
        st.write("Starting with a new count")
with col2:
    if st.button("Next Pitch"):
        st.write("Simulating next pitch")