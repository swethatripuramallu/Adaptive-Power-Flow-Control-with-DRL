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

mpc = loadcase(case14);
% for each training step 
Pload1 = 47.8;
Qload1 = -3.9;
Pload2 = 7.6;
Qload2 = 1.6; 
Pload3 = 29.5;
Qload3 = 16.6;
Pload4 = 9; 
Qload4 = 5.8;
Pload5 = 3.5;
Qload5 = 1.8;
Pload6 = 6.1;
Qload6 = 1.6;
Pload7 = 13.5;
Qload7 = 5.8;
Pload8 = 14.9;
Qload8 = 5; 
Ppv = 5; 
Pw = 1; 
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
Pbss = 1; 
Qbss = 0.5; 
Qw = 0.5; 
Qpv1 = 7; 
Qpv2 = 7; 
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

