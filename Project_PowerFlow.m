% Project 595
clear
clc
% Parameters for distribution network 
% For Solar Panel System (on bus 2 and 3, row 2 and 3 on gen matrix) 
% Q_pv_max = 6.87 MW (for both systems) 
% Q_pv_min = 0 MW
% For Wind Turbine (on bus 8, row 5 on gen matrix)
% Q_w_max = 0.968 MW
% Q_w_min = 0
% For battery (on bus 6, row 4 on gen matrix) 
% P_bss_max = 2
% P_bss_min = -2
% Q_bss_max = 0.968 
% Q_bss_min = 0 
% charge/discharge efficiency, nu_ch = 0.98
% E_max = 0.9*4 MWh; 
% E_min = 0.2*4 MWh

% Voltage Limits (at each bus) 
% V_min = 1.06; 
% V_max = 0.94; 

addpath('C:\Users\sweth\Downloads\matpower8.0\matpower8.0')
savepath;

mpc = loadcase('case14');
% for each training step 
Pload1 = 15.2900000000000;
Qload1 = 3.83203500000000;
Pload2 = 6.89200000000000;
Qload2 = 3.33794794700000; 
Pload3 = 4.91600000000000;
Qload3 = 1.94292752200000;
Pload4 = 5.04000000000000; 
Qload4 = 2.29629184000000;
Pload5 = 4.16300000000000;
Qload5 = 14.0960000000000;
Pload6 = 14.0960000000000;
Qload6 = 6.42232733600000;
Pload7 = 17.0810000000000;
Qload7 = 6.75084316500000;
Pload8 = 7.13600000000000;
Qload8 = 2.82032766400000; 
			
Ppv = 0; 
Pw = 0; 
% obtain generation from solar and wind and load for all buses 
mpc.bus(4,3:4) = [Pload1 Qload1]; 
mpc.bus(5,3:4) = [Pload2 Qload2];
mpc.bus(9,3:4) = [Pload3 Qload3]; 
mpc.bus(10,3:4) = [Pload4 Qload4]; 
mpc.bus(11,3:4) = [Pload5 Qload5]; 
mpc.bus(12,3:4) = [Pload6 Qload6]; 
mpc.bus(13,3:4) = [Pload7 Qload7];
mpc.bus(14,3:4) = [Pload8 Qload8]; 
mpc.gen(2,2) = Ppv; 
mpc.gen(3,2) = Ppv; 
mpc.gen(5,2) = Pw; 
% Obtain actions from agent 
Pbss = -0.5263738036155701; 
Qbss = 0.22232136130332947; 
Qw = 0.1125740334391594; 
Qpv1 = -0.1310410499572754; 
Qpv2 = 0.813401997089386; 


% Update power flow casefile with obtained actions 
mpc.gen(4,2) = Pbss; 
mpc.gen(4,3) = Qbss; 
mpc.gen(2,3) = Qpv1; 
mpc.gen(3,3) = Qpv2; 
mpc.gen(5,3) = Qw; 
% results 
results = runpf(mpc); 
losses = get_losses(results); 
system_loss = sum((losses)); 
 
bus_voltages = mpc.bus(:,8); 
bus_voltage_angle = mpc.bus(:,9); 

