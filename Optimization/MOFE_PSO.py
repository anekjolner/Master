# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 22:21:04 2022

@author: erbet&carine
"""
import numpy as np
import math
import time
import lhsmdu
import copy


class Particle:
    def __init__(self, name, x, v, obj, cons, isFeasible,objIsEvaluated,nConsEval,nObjEvals):
        self.name = name
        self.x = x
        self.v = v
        self.obj = obj
        self.cons = cons
        self.isFeasible = isFeasible
        self.objIsEvaluated = objIsEvaluated
        self.nConsEval = nConsEval
        self.nObjEvals = nObjEvals
    # END 
# END 
class MOFE_PSO:
    
    def RUN(options,GlobalVars):
        if options.useInputDecoder:
            return options.InputDecoder 
        # END 
        nVars = len(options.lBound)
        if nVars != len(options.uBound):
            raise ValueError('Incompattible bounds provided')
        # END 
          # Some counters
        GlobalVars.nObjEvals = np.array([0])
        GlobalVars.nConsEvals = np.array([0])
        options.Iterator = np.array([0])
        
        
        # Initialize particle array
        results = type('',(),{})()
        results.Particle = type('',(),{})()
        results.Particle.nonDom = [] # np.array([])
        results.Particle.pos = [] #np.array([])
        results.Particle.v = np.zeros([options.swarmSize,options.Nvar]) # Array para alocação das velocidades das partículas
        results.nonDom = [] # Lista para alocação das partículas do pareto
        results.AllFisPart = [] # Lista para alocação de todas as partículas viáveis do PSO
        results.TotalTimer= np.zeros([options.maxIter])
        results.PartTimer = np.zeros([options.maxIter,options.swarmSize])
        results.conSens = []
        results.violated = []
        results.options = type('',(),{})()
        results.nObjEvals = []
        results.nConsEvals = []
        for i in range(options.swarmSize):
            results.Particle.nonDom.append(np.array([]))
            results.Particle.pos.append(Particle("Particle_"+str(i),[],[],[],[],[],[],[],[]))
        # END  
          
        # initialize violated matrix: [swarmSize x nCons] sized boolean 
        # matrix containing which particle's current position violates which 
        # constraint, violates if 1
        violated = np.ones((options.swarmSize,options.nCons),dtype=bool)
          
        #Loop the initialization procedure until there is at least one
        #feasible particle for each of the constraints
        firstTry = True
       
        while np.any(np.all(violated,axis=0)) or (options.nCons == 0 and firstTry):
            firstTry = False
            Iteration = 0
            if options.verbose>=1:
                print('Initializing particles\n')
            # generate random positions for the particles
            
            random_num = np.array(lhsmdu.sample(options.swarmSize,options.Nvar))
            
            for i in range(options.swarmSize):
                pos = Particle("Particle_"+str(i),[],[],[],[],[],[],[],[])
                pos.x = options.lBound+ random_num[i,:]*(options.uBound-options.lBound)
                
                (pos,options,GlobalVars) = MOFE_PSO.evalPos(pos,options,GlobalVars)          
                
                (options,GlobalVars,violated,results) = MOFE_PSO.setParticlePos(pos,i,options,GlobalVars,violated,results,Iteration)
                
                results.Particle.v[i,:] = options.velocityInitializationFactor*(-(options.uBound-options.lBound) + np.random.rand(1,options.Nvar)*(2*(options.uBound-options.lBound)))
     
            # END 
        # END 
          
        
    
        if options.verbose>=1:
            print('Calculating constraint sensitivities\n')
            
        # END 
        if options.nCons>=1:
            consSens = MOFE_PSO.calculateConstraintSensitivities(options,results,GlobalVars)
        else:
            consSens = []
        # END 
        # initialize number of times constraints are activated by each particle
        ac_nSel = np.zeros([options.swarmSize,options.nCons])  #activated constraint selection count
        # The main optimization loop  
        
        TotalTimer = time.time()
        for m in range(options.maxIter):
            Iteration = m+1
            if options.verbose>=1:
                print("Iteration "+str(m)+" started.")
                StartiterTimer = copy.copy(time.time())
                progress = (m-1) / (options.maxIter - 1);
                C0 = options.initialC0 + progress * (options.finalC0 - options.initialC0);
    
            #iterate on particles    
            for i in range(options.swarmSize):
                # particle's current position
                startPartTimer = copy.copy(time.time())
                curPos = results.Particle.pos[i]
                
                if curPos.isFeasible:
                    # feasible particle behavior
                    # randomly select a leader from gbest
                    index = np.random.randint(0,len(results.nonDom))
                    globalGuide = results.nonDom[index]
                    index = np.random.randint(0,len(results.Particle.nonDom[i]))
                    localGuide = results.Particle.nonDom[i][index]
                    
                    #calculate velocity
                    v = C0 * results.Particle.v[i] + options.C1 * np.random.rand(1,options.Nvar) * (globalGuide.x - curPos.x) + options.C2 * np.random.rand(1,options.Nvar) * (localGuide.x - curPos.x)
                    results.AllFisPart = np.append(results.AllFisPart, localGuide)
                else:
                    #infeasible particle behavior
                    #constraints violated by this particle
                    
                    localViolated = violated[i,:]
    
                    #indexes of violated constraints
                    localViolatedIndexes = np.argwhere(localViolated==True)
                    #times these violated constraints have been selected as AC
                    nSelLocalViolated = ac_nSel[i,:]
                    nSelLocalViolated = nSelLocalViolated[localViolated]
    
                    # the least visited violated constraints
                    
                    leastVisitedLocalViolatedIndexes = localViolatedIndexes[nSelLocalViolated == np.min(nSelLocalViolated)]
                    
                    
                    #global violation counts of constraints to be used as priority
                    nViol = sum(violated)
                    
                    #global violation counts of the least visited violated constraints
                    nViolLeastVisitedLocalViolated = nViol[leastVisitedLocalViolatedIndexes]
                    
                    #highest priority among the least visited violated constraints
                    highestPriorityLeastVisitedLocalViolatedIndexes = leastVisitedLocalViolatedIndexes[nViolLeastVisitedLocalViolated == nViolLeastVisitedLocalViolated.max()]
                    
                    # randomly select a constraint as the activated constraint from
                    # the highest priority list
                    
                    index = np.random.randint(0,len(highestPriorityLeastVisitedLocalViolatedIndexes))
                    activatedConstraint = highestPriorityLeastVisitedLocalViolatedIndexes[index]
                    
                    # increase the ac selection count
                    
                    ac_nSel[i,activatedConstraint] = ac_nSel[i,activatedConstraint] + 1;
    
                    # populate particles that don't violate the activated
                    # constraint
                    aux = violated[:,activatedConstraint]
    
                    acFeasibleParticleIndexes = np.argwhere(aux==False)
                    
                    # randomly select one
                    
                    index = np.random.randint(0,len(acFeasibleParticleIndexes))
    
    
                    globalGuide = results.Particle.pos[acFeasibleParticleIndexes[index][0]]
                    
                    #calculate velocity
                    v = C0 *results.Particle.v[i]+ options.C1 * np.random.rand(1,options.Nvar)* (globalGuide.x - curPos.x)
                    v = v * consSens[activatedConstraint,:]
                    
                #END
                
                # Set particle's velocity
                
                results.Particle.v[i] = np.nan_to_num(v[0])
                
                # Create a candidate point
    
                x_c = curPos.x+v[0]
                
                # Enforce variables limits
                
                (x_c,isEnforced) = MOFE_PSO.enforceVariableLimits(x_c,curPos.x,options)
    
                if isEnforced:
                    results.Particle.v[i] = np.zeros(options.Nvar)
                #END
                
                pos_c = MOFE_PSO.sameOrNewPos(x_c,options,GlobalVars,curPos)
    
                
                (newPos, isSearched) = MOFE_PSO.virtualBoundarySearch(pos_c, curPos,options)
                
                
                if isSearched:
                    results.Particle.v[i] = np.zeros(options.Nvar)
                
                
                (options, GlobalVars, violated, results) = MOFE_PSO.setParticlePos(newPos, i, options, GlobalVars, violated, results,Iteration)
        
                results.PartTimer[m,i] = copy.copy(time.time()) - startPartTimer
            #END
            results.TotalTimer[m] = copy.copy(time.time()) - StartiterTimer
            if options.verbose>=1:
                print("Iteration "+str(m)+" finished in "+str(copy.copy(time.time()) - StartiterTimer)+"s.")
        #END
        print("Optimization solved in: "+str(time.time() - TotalTimer)+" s.")
        return results, GlobalVars, options
    # END 
    
    def virtualBoundarySearch(pos_try, pos_p,options):
        # record constraints that are already satisfied by the current
        # position
        
        alreadySatisfied = (pos_p.cons <= 0)
    
        #if there are any new violations
        if np.any(np.logical_and(alreadySatisfied, pos_p.cons>0)):
            #some already satisfied constraints are violated. Therefore, we
            #start the boundary search.
            pos_n = pos_try
            inTolerances = False
            while not inTolerances:
                if any(pos_p.cons[np.logical_and(alreadySatisfied, pos_n.cons>0)]==0):
                    
                    q = math.inf
                    x_try = pos_p.x
                    pos_try = pos_p
                else:
                    q = max(pos_n.cons[np.logical_and(alreadySatisfied, pos_n.cons>0)]/ (-pos_p.cons[np.logical_and(alreadySatisfied, pos_n.cons>0)]))
                    # generate new trial position
                    # Garantir que, se o ponto testado for na restrição, o peso w será inf.
                    if np.isnan(q) or np.isinf(q):
                      q = math.inf
                    x_try = pos_p.x + (pos_n.x - pos_p.x) / (1+q)
                    pos_try = MOFE_PSO.sameOrNewPos(x_try,pos_n,pos_p)
                #END
                # calculate distance to closer edge
                if q>=1:
                    distanceFactor = abs(x_try - pos_p.x)/ (options.uBound - options.lBound)
                else:
                    distanceFactor = abs(x_try - pos_n.x)/ (options.uBound - options.lBound)
                #END
                
                if all(distanceFactor<options.boundaryTolerance):
                    inTolerances = True
                else:
                    inTolerances = False
                #END
                
                if alreadySatisfied.any() and any(pos_try.cons > 0):
                    # if trial point is not virtually feasible it replaces
                    #
                    pos_n = pos_try
                else:
                    pos_p = pos_try
                #END
                              
            #END
            # Return the virtually feasible edge as the output
            pos_out = pos_p()
            isSearched = True
                        
        pos_out = pos_try
        isSearched = False
        return pos_out, isSearched
    #END
    
    
    def sameOrNewPos(x,options,GlobalVars,*args):
        
        varargin = args
    
        nargin = len(varargin)
        
        for p in range(nargin):
            oldPos = varargin[p]
            
            if all(x == oldPos.x):
                pos = oldPos
                pos.x = x
                
                return pos
        #END  
        pos = Particle("Particle_",[],[],[],[],[],[],[],[])
        pos.x = x
        (pos, options, GlobalVars) = MOFE_PSO.evalPos(pos,options,GlobalVars)
        return pos
    #END
    
    
    def evalPos(pos,options,GlobalVars):
        
        if options.nCons>=1:
            pos.cons = options.ConsFunc(pos.x)
        else:
            pos.cons = 0
        # END 
        GlobalVars.nConsEvals =  GlobalVars.nConsEvals +1
          
        if np.all(pos.cons<=0):
            
            pos.isFeasible = True
            pos.obj = []
            pos.objIsEvaluated = False
        else:
            pos.isFeasible = False
            pos.obj = []
        # END 
        
        pos.nConsEvals = GlobalVars.nConsEvals
          
        return pos, options, GlobalVars
    # END 
    
    
    def setParticlePos(pos,ParticleID,options,GlobalVars,violated,results,Iteration):
        
        if pos.isFeasible and not pos.objIsEvaluated:
            pos.obj = options.ObjFun(pos.x)
            GlobalVars.nObjEvals = GlobalVars.nObjEvals + 1
            pos.nObjEvals = GlobalVars.nObjEvals
        # END 
        results.Particle.pos[ParticleID] = pos
        
        # update violation counts
        
        if options.nCons >=1:
            violated[ParticleID,:] = np.transpose(pos.cons>0)
        # END 
    
        if Iteration==0:
            results.Particle.nonDom[ParticleID] = np.array([])
        
        if pos.isFeasible==1:
            (newSet, hasChanged) = MOFE_PSO.updateNonDominatedSet(results.Particle.nonDom[ParticleID],pos,options)
    
            if hasChanged==1:
                
                results.Particle.nonDom[ParticleID] = newSet
                
                # Update the matrix results.nomDom
                
                (newGlobalSet, globalHasChanged) = MOFE_PSO.updateNonDominatedSet(results.nonDom,pos,options)
    
                if globalHasChanged:
                    results.nonDom = newGlobalSet
                # END
            # END 
        # END 
        return options, GlobalVars, violated, results
    # END 
    
    
    def updateNonDominatedSet(oldSet,pos,options):
        
        # oldSet is a list that has all particles
        isNonDom = True
      
    
        dominated = np.zeros(len(oldSet),dtype=bool)
        
    
        for k in range(len(oldSet)):
            #check if new position is dominated by any of the members
            #of the particle's non dominated set
            if MOFE_PSO.checkDomination(oldSet[k].obj,pos.obj):
                isNonDom = False
    
                break
            # END 
            #We also need to check if the new position dominates any of
            #the existing ones
            if MOFE_PSO.checkDomination(pos.obj,oldSet[k].obj):
                dominated[k] = True
            # END 
        
        if isNonDom:
             #check if the new point is practically same as any already
             #existing member of the set
             isSame = False
             for k in range(len(oldSet)):
                 if all(pos.x == oldSet[k].x):
                     
                     isSame = True
    
                     break
                 # END 
             # END 
             
             if not isSame:
                 if len(oldSet)==0:
                     newSet = np.array([pos])
                     hasChanged = 1
                 else:
                     newSet = np.append(oldSet[~dominated], pos)
                     hasChanged = 1
             else:
                 hasChanged = 0
                 newSet = oldSet
             # END 
             
        else:
            hasChanged = 0
            newSet = oldSet
            
        # END 
        return newSet, hasChanged
    # END 
    
    # ok
    def checkDomination(testObj,againstObj):
        result = all(testObj<=againstObj) and any(testObj < againstObj)
        return result
    # END 
    
    
    def calculateConstraintSensitivities(options,results,GlobalVars):
        cs = np.zeros([options.nCons,options.Nvar],dtype=bool)
        increment = options.incrementFactor * (options.uBound - options.lBound)
        
        for p in range(options.swarmSize):
            pos = Particle("Particle_"+str(p),[],[],[],[],[],[],[],[])
            pos.x = results.Particle.pos[p].x
            pos.cons = results.Particle.pos[p].cons
            for d in range(options.Nvar):
                trialX = Particle("Particle_"+str(p),[],[],[],[],[],[],[],[])
                trialX.x = pos.x
                trialX.x[d] = trialX.x[d] + increment[d]
                if not trialX.x[d]>options.uBound[d]:
                    (trialPos, options, GlobalVars) = MOFE_PSO.evalPos(trialX, options, GlobalVars)
                    s = (trialPos.cons != pos.cons)
                    for jj in range(len(s)):
                        cs[jj,d] = (cs[jj,d] or s[jj])
                    # END 
                # END
                       
                trialX.x = pos.x
                trialX.x[d] = trialX.x[d] - increment[d]
                if not trialX.x[d]<options.lBound[d]:
                    (trialPos, options, GlobalVars) = MOFE_PSO.evalPos(trialX, options, GlobalVars)
                    s = (trialPos.cons != pos.cons)
                    for jj in range(len(s)):
                        cs[jj,d] = (cs[jj,d] or s[jj])
                    # END
                # END 
            # END
        # END 
        return cs
    # END 
    
    def enforceVariableLimits(x_try,x_vf,options):
        isEnforced = False
    
        if (x_try < options.lBound).any() or (x_try > options.uBound).any():
            isEnforced = True
    
            w = np.max([np.max( (x_try - options.uBound) / (options.uBound - x_vf) ),np.max((options.lBound - x_try) / (x_vf - options.lBound) ) ])
            
            # Garantir que, se o ponto testado for na restrição, o peso w será inf.
            if np.isnan(w) or np.isinf(w):
                w = math.inf
            x_try = x_vf + (x_try - x_vf) / (1+w)
            #Limits might still be violated due to precision errors
            #following 2 lines are to prevent this
            x_try[x_try<options.lBound] = options.lBound[x_try<options.lBound]
            x_try[x_try > options.uBound] = options.uBound[x_try > options.uBound]
        x_out = x_try
        return x_out,isEnforced
    #END    