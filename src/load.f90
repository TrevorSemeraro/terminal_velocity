PROGRAM load_sample_data
    IMPLICIT NONE

    TYPE :: test_record
        real(kind(0d0)) :: DIAMETER
        real(kind(0d0)) :: TEMPERATURE
        real(kind(0d0)) :: PRESSURE
        real(kind(0d0)) :: VELOCITY
        real(kind(0d0)) :: DENSITY
        real(kind(0d0)) :: DYNAMIC_VISCOSITY
    END TYPE test_record

    TYPE :: results_record
        real(kind(0d0)) :: beard
        real(kind(0d0)) :: simmel
        real(kind(0d0)) :: two_stage
        real(kind(0d0)) :: two_stage_MINI
    END TYPE results_record

    TYPE(test_record), DIMENSION(1000000) :: sample_data
    TYPE(results_record), DIMENSION(1000000) :: results

    INTEGER :: i, ios
    real(kind(0d0)) :: beard_velocity, computed_simmel_velocity, &
        computed_2stage_model_velocity, computed_2stage_model_MINI_velocity
    
    ! Timing variables
    real(kind(0d0)) :: start_time, end_time, iteration_time
    real(kind(0d0)) :: beard_loop_time, simmel_loop_time, twostage_loop_time, twostage_MINI_loop_time
    INTEGER :: clock_start, clock_end, clock_rate, clock_max
    real(kind(0d0)) :: total_time
    real(kind(0d0)) :: checksum
    
    INTEGER :: j, n_repeats

    ! Open the file for reading
    OPEN(UNIT=18, FILE='../data/test_data.csv', STATUS='OLD', ACTION='READ', IOSTAT=ios)
    IF (ios /= 0) THEN
        PRINT *, 'Error opening file.'
        STOP
    END IF

    do i = 1, 1000000
        read(18,*) sample_data(i)
    end do

    CLOSE(18)
    PRINT *, 'Finished reading file.'

    n_repeats = 100

    ! Get system clock rate for high precision timing
    CALL SYSTEM_CLOCK(COUNT_RATE=clock_rate, COUNT_MAX=clock_max)
    PRINT *, 'System clock precision: ', 1.0D0/REAL(clock_rate), ' seconds'

    ! Beard terminal velocity timing with high precision    
    PRINT *, 'Timing Beard model...'
    total_time = 0.0D0
    DO i = 1, 1000000
        beard_velocity = beard_terminal_velocity(sample_data(i)%DIAMETER, &
            sample_data(i)%TEMPERATURE, &
            sample_data(i)%PRESSURE)
        results(i)%beard = beard_velocity
    END DO
    CALL SYSTEM_CLOCK(clock_start)
    DO j=1, n_repeats
        checksum = 0.0
        DO i = 1, 1000000
            checksum = checksum + beard_terminal_velocity(sample_data(i)%DIAMETER, &
                sample_data(i)%TEMPERATURE, &
                sample_data(i)%PRESSURE)
        END DO
    END DO
    CALL SYSTEM_CLOCK(clock_end)
    total_time = REAL(clock_end - clock_start) / REAL(clock_rate)
    beard_loop_time = total_time / DBLE(n_repeats)
    if (checksum == 0.0) print *, 'Checksum is zero (impossible)'

    ! Simmel terminal velocity timing and saving results
    PRINT *, 'Timing Simmel model...'
    total_time = 0.0D0
    DO i = 1, 1000000
        computed_simmel_velocity = simmel_velocity(sample_data(i)%DIAMETER, &
            sample_data(i)%TEMPERATURE, &
            sample_data(i)%PRESSURE)
        results(i)%simmel = computed_simmel_velocity
    END DO
    CALL SYSTEM_CLOCK(clock_start)
    DO j=1, n_repeats
        checksum = 0.0
        DO i = 1, 1000000
            checksum = checksum + simmel_velocity(sample_data(i)%DIAMETER, &
                sample_data(i)%TEMPERATURE, &
                sample_data(i)%PRESSURE)
        END DO
    END DO
    CALL SYSTEM_CLOCK(clock_end)
    total_time = REAL(clock_end - clock_start) / REAL(clock_rate)
    simmel_loop_time = total_time / DBLE(n_repeats)
    if (checksum == 0.0) print *, 'Checksum is zero (impossible)'

    ! Two-stage Model terminal velocity timing and saving results
    PRINT *, 'Timing Two-Stage model...'
    total_time = 0.0D0
    DO i = 1, 1000000
        computed_2stage_model_velocity = two_stage_model(sample_data(i)%DIAMETER, &
            sample_data(i)%TEMPERATURE, &
            sample_data(i)%DENSITY, &
            sample_data(i)%DYNAMIC_VISCOSITY)
        results(i)%two_stage = computed_2stage_model_velocity
    END DO
    CALL SYSTEM_CLOCK(clock_start)
    DO j=1, n_repeats
        checksum = 0.0
        DO i = 1, 1000000
            checksum = checksum + two_stage_model(sample_data(i)%DIAMETER, &
                sample_data(i)%TEMPERATURE, &
                sample_data(i)%DENSITY, &
                sample_data(i)%DYNAMIC_VISCOSITY)
        END DO
    END DO
    CALL SYSTEM_CLOCK(clock_end)
    total_time = REAL(clock_end - clock_start) / REAL(clock_rate)
    twostage_loop_time = total_time / DBLE(n_repeats)
    if (checksum == 0.0) print *, 'Checksum is zero (impossible)'

    ! Two-stage_MINI Model terminal velocity timing and saving results
    PRINT *, 'Timing Two-Stage model...'
    total_time = 0.0D0
    DO i = 1, 1000000
        computed_2stage_model_MINI_velocity = two_stage_model_MINI(sample_data(i)%DIAMETER, &
            sample_data(i)%TEMPERATURE, &
            sample_data(i)%DENSITY, &
            sample_data(i)%DYNAMIC_VISCOSITY)
        results(i)%two_stage_MINI = computed_2stage_model_MINI_velocity
    END DO
    CALL SYSTEM_CLOCK(clock_start)
    DO j=1, n_repeats
        checksum = 0.0
        DO i = 1, 1000000
            checksum = checksum + two_stage_model_MINI(sample_data(i)%DIAMETER, &
                sample_data(i)%TEMPERATURE, &
                sample_data(i)%DENSITY, &
                sample_data(i)%DYNAMIC_VISCOSITY)
        END DO
    END DO
    CALL SYSTEM_CLOCK(clock_end)
    total_time = REAL(clock_end - clock_start) / REAL(clock_rate)
    twostage_MINI_loop_time = total_time / DBLE(n_repeats)
    if (checksum == 0.0) print *, 'Checksum is zero (impossible)'
    
    ! Write results to CSV file
    PRINT *, ''
    PRINT *, 'Writing results to terminal_velocity_results.csv...'
    OPEN(UNIT=19, FILE='../data/terminal_velocity_results.csv', STATUS='REPLACE', ACTION='WRITE', IOSTAT=ios)
    IF (ios /= 0) THEN
        PRINT *, 'Error creating output file.'
        STOP
    END IF

    ! Write CSV header
    WRITE(19, '(A)') 'DIAMETER,TEMPERATURE,PRESSURE,ORIGINAL_VELOCITY,BEARD,SIMMEL,TWO_STAGE,TWO_STAGE_MINI'
    
    ! Write data rows
    DO i = 1, 1000000
        WRITE(19, '(E15.8,",",E15.8,",",E15.8,",",E15.8,",",E15.8,",",E15.8,",",E15.8,",",E15.8)') &
            sample_data(i)%DIAMETER, &
            sample_data(i)%TEMPERATURE, &
            sample_data(i)%PRESSURE, &
            sample_data(i)%VELOCITY, &
            results(i)%beard, &
            results(i)%simmel, &
            results(i)%two_stage, &
            results(i)%two_stage_MINI
    END DO
    
    CLOSE(19)
    PRINT *, 'Results successfully written to terminal_velocity_results.csv'

    ! Write timing results to CSV file
    PRINT *, ''
    PRINT *, 'Writing timing results to model_timing_results.csv...'
    OPEN(UNIT=20, FILE='../data/model_timing_results.csv', STATUS='REPLACE', ACTION='WRITE', IOSTAT=ios)
    IF (ios /= 0) THEN
        PRINT *, 'Error creating timing output file.'
        STOP
    END IF

    ! Write CSV header
    WRITE(20, '(A)') 'MODEL_NAME,AVERAGE_EXECUTION_TIME_SECONDS,ITERATIONS'
    
    ! Write timing data rows
    WRITE(20, '(A,",",E15.8,",",I0)') 'BEARD', beard_loop_time, n_repeats
    WRITE(20, '(A,",",E15.8,",",I0)') 'SIMMEL', simmel_loop_time, n_repeats
    WRITE(20, '(A,",",E15.8,",",I0)') 'TWO_STAGE', twostage_loop_time, n_repeats
    WRITE(20, '(A,",",E15.8,",",I0)') 'TWO_STAGE_MINI', twostage_MINI_loop_time, n_repeats
    
    CLOSE(20)
    PRINT *, 'Timing results successfully written to model_timing_results.csv'
    
    ! Print timing summary to console
    PRINT *, ''
    PRINT *, '========== TIMING SUMMARY =========='
    PRINT *, 'Model                | Avg Time (s) | Iterations'
    PRINT *, '---------------------|--------------|-----------'
    WRITE(*, '(A,"|",E13.6,"|",I10)') 'Beard              ', beard_loop_time, n_repeats
    WRITE(*, '(A,"|",E13.6,"|",I10)') 'Simmel             ', simmel_loop_time, n_repeats  
    WRITE(*, '(A,"|",E13.6,"|",I10)') 'Two-Stage          ', twostage_loop_time, n_repeats
    WRITE(*, '(A,"|",E13.6,"|",I10)') 'Two-Stage_MINI          ', twostage_MINI_loop_time, n_repeats
    PRINT *, '===================================='

CONTAINS

    ! 
    ! TWO STAGE MODEL COPY PASTE
    ! 



PURE ELEMENTAL FUNCTION min_ref(x1) result(res)
        IMPLICIT NONE
            real(kind(0d0)), INTENT(IN) :: x1
        real(kind(0d0)) :: res
        res = (24.577503098687846d0*x1 - 9.833516965546338d0/( &
      2.683145025224074d0 - 0.7198048281825444d0/(-abs( &
      0.3378666219187161d0 - 2.529907013476554d0/(0.5109283913335468d0 &
      - 0.0019830203447073792d0/x1)) + 0.09022045044656467d0/( &
      -38.204695171459442d0*x1/(-0.21779415215862336d0)**2 + &
      0.15999162529675903d0)) + 0.0017739234165219686d0/x1))**2
    END FUNCTION min_ref
PURE ELEMENTAL FUNCTION min_corr(x1, x2, x3) result(res)
        IMPLICIT NONE
            real(kind(0d0)), INTENT(IN) :: x1
    real(kind(0d0)), INTENT(IN) :: x2
    real(kind(0d0)), INTENT(IN) :: x3
        real(kind(0d0)) :: res
        res = abs(x2**2/(0.052402248617831146d0*abs(x2) + abs(x2 + &
      9.569028026412478d0*(1869.6799115010711d0 - 8.0512265548176943d0* &
      (x2 - 42.873686952706809d0)/x3)/(x2 + 0.065921659380834839d0/(x1* &
      x3)) + (0.0062625065005005198d0*x2 - 1.4470731375012897d0)/(x1 + &
      0.018806424953245368d0)) - 33.132060806498476d0)**2)
    END FUNCTION min_corr



    ! 
    ! 
    ! 

    FUNCTION two_stage_model_MINI(x0, x1, x2, x3) RESULT(terminal_velocity)
        IMPLICIT NONE
        real(kind(0d0)), INTENT(IN) :: x0, x1, x2, x3
        real(kind(0d0)) :: terminal_velocity

        terminal_velocity = min_ref(x0) * min_corr(x0, x1, x2)
    END FUNCTION two_stage_model_MINI



    ! 
    ! TWO STAGE MODEL COPY PASTE
    ! 


PURE ELEMENTAL FUNCTION ref(x1) result(res)
        IMPLICIT NONE
            real(kind(0d0)), INTENT(IN) :: x1
        real(kind(0d0)) :: res
        res = (-9.63350690045338d0/(-2.7215338890170087d0 + 0.6267721824987109d0 &
      /(-abs(216.6249971887023d0*x1**2/((-2.1260595592984781d0*x1 - &
      0.031181841922384247d0)*(-2.0d0*x1 - 0.04107186782823692d0)) + &
      0.38117379931425016d0 + 2.1823723602520495d0/( &
      -0.5130823865258124d0 + 0.0018211419393372632d0/x1)) + &
      0.09025582106317721d0/(33.08021237549337d0*x1/( &
      24.143797670270116d0*x1 - 0.04941794322271916d0) + &
      0.15579864209390054d0)) - 0.0017453161714130038d0/x1))**2
    END FUNCTION ref
PURE ELEMENTAL FUNCTION corr(x1, x2, x3) result(res)
        IMPLICIT NONE
            real(kind(0d0)), INTENT(IN) :: x1
    real(kind(0d0)), INTENT(IN) :: x2
    real(kind(0d0)), INTENT(IN) :: x3
        real(kind(0d0)) :: res
        res = sqrt((-x2/(-1.355900186947505d0*x2 + 52.855078242329108d0*x2/( &
      -453.47519177667665d0*x1*x2*x3/abs(5594025.1837800374d0*x1**3 - &
      0.05528710792205269d0*x2 + 25.071331867730663d0) + x2*x3 + &
      5.9302179415776721d-5*x3/(x1**2*abs(5594025.1837800374d0*x1**3 - &
      0.05528710792205269d0*x2 + 25.071331867730663d0))) + &
      8.5108364985610056d0*x2/(-453.47519177667665d0*x1*x2/abs( &
      5594025.1837800374d0*x1**3 - 0.05528710792205269d0*x2 + &
      25.071331867730663d0) + x2 + 5.9302179415776721d-5/(x1**2*abs( &
      5594025.1837800374d0*x1**3 - 0.05528710792205269d0*x2 + &
      25.071331867730663d0))) - 1883.3828630902532d0*x3/( &
      -453.47519177667665d0*x1*x2/abs(5594025.1837800374d0*x1**3 - &
      0.05528710792205269d0*x2 + 25.071331867730663d0) + x2 + &
      5.9302179415776721d-5/(x1**2*abs(5594025.1837800374d0*x1**3 - &
      0.05528710792205269d0*x2 + 25.071331867730663d0))) + &
      101.64243612640218d0 - 2361.2542199276047d0/( &
      -453.47519177667665d0*x1*x2*x3/abs(5594025.1837800374d0*x1**3 - &
      0.05528710792205269d0*x2 + 25.071331867730663d0) + x2*x3 + &
      5.9302179415776721d-5*x3/(x1**2*abs(5594025.1837800374d0*x1**3 - &
      0.05528710792205269d0*x2 + 25.071331867730663d0))) - &
      12076.637715147497d0/(-453.47519177667665d0*x1*x2/abs( &
      5594025.1837800374d0*x1**3 - 0.05528710792205269d0*x2 + &
      25.071331867730663d0) + x2 + 5.9302179415776721d-5/(x1**2*abs( &
      5594025.1837800374d0*x1**3 - 0.05528710792205269d0*x2 + &
      25.071331867730663d0)))))**0.94274485865591751d0*((x2/( &
      1.355900186947505d0*x2 - 52.855078242329108d0*x2/( &
      -453.47519177667665d0*x1*x2*x3/abs(5594025.1837800374d0*x1**3 - &
      0.05528710792205269d0*x2 + 25.071331867730663d0) + x2*x3 + &
      5.9302179415776721d-5*x3/(x1**2*abs(5594025.1837800374d0*x1**3 - &
      0.05528710792205269d0*x2 + 25.071331867730663d0))) - &
      8.5108364985610056d0*x2/(-453.47519177667665d0*x1*x2/abs( &
      5594025.1837800374d0*x1**3 - 0.05528710792205269d0*x2 + &
      25.071331867730663d0) + x2 + 5.9302179415776721d-5/(x1**2*abs( &
      5594025.1837800374d0*x1**3 - 0.05528710792205269d0*x2 + &
      25.071331867730663d0))) + 1883.3828630902532d0*x3/( &
      -453.47519177667665d0*x1*x2/abs(5594025.1837800374d0*x1**3 - &
      0.05528710792205269d0*x2 + 25.071331867730663d0) + x2 + &
      5.9302179415776721d-5/(x1**2*abs(5594025.1837800374d0*x1**3 - &
      0.05528710792205269d0*x2 + 25.071331867730663d0))) - &
      101.64243612640218d0 + 2361.2542199276047d0/( &
      -453.47519177667665d0*x1*x2*x3/abs(5594025.1837800374d0*x1**3 - &
      0.05528710792205269d0*x2 + 25.071331867730663d0) + x2*x3 + &
      5.9302179415776721d-5*x3/(x1**2*abs(5594025.1837800374d0*x1**3 - &
      0.05528710792205269d0*x2 + 25.071331867730663d0))) + &
      12076.637715147497d0/(-453.47519177667665d0*x1*x2/abs( &
      5594025.1837800374d0*x1**3 - 0.05528710792205269d0*x2 + &
      25.071331867730663d0) + x2 + 5.9302179415776721d-5/(x1**2*abs( &
      5594025.1837800374d0*x1**3 - 0.05528710792205269d0*x2 + &
      25.071331867730663d0)))))**0.94274485865591751d0))**2
    END FUNCTION corr



    ! 
    ! 
    ! 

    FUNCTION two_stage_model(x0, x1, x2, x3) RESULT(terminal_velocity)
        IMPLICIT NONE
        real(kind(0d0)), INTENT(IN) :: x0, x1, x2, x3
        real(kind(0d0)) :: terminal_velocity

        terminal_velocity = ref(x0) * corr(x0, x1, x2)
    END FUNCTION two_stage_model

    FUNCTION simmel_velocity(diameter, temperature, pressure) RESULT(terminal_velocity)
        IMPLICIT NONE
        real(kind(0d0)), INTENT(IN) :: diameter, temperature, pressure
        real(kind(0d0)) :: terminal_velocity
        
        ! Constants
        real(kind(0d0)), PARAMETER :: R_air = 287.05
        real(kind(0d0)), PARAMETER :: rho_ref = 1.2
        real(kind(0d0)), PARAMETER :: pi = 3.14159265359
        
        ! Coefficients from Table 2 (Simmel et al.)
        real(kind(0d0)), PARAMETER :: alphas(4) = (/ 4.5795E5, 4.962E3, 1.732E3, 9.17E2 /)  ! cm/s
        real(kind(0d0)), PARAMETER :: betas(4) = (/ 2.0/3.0, 1.0/3.0, 1.0/6.0, 0.0 /)
        
        ! Local variables
        real(kind(0d0)) :: rho_air, air_density_correction
        real(kind(0d0)) :: d_um, radius_m, volume_m3, mass_g
        real(kind(0d0)) :: v_t_cm_s
        
        ! Calculate air density and correction factor
        rho_air = pressure / (temperature * R_air)
        air_density_correction = (rho_air / rho_ref) ** 0.54
        
        ! Convert diameter to micrometers
        d_um = diameter * 1.0E6
        
        ! Calculate droplet mass in grams
        radius_m = diameter / 2.0
        volume_m3 = (4.0/3.0) * pi * radius_m**3
        mass_g = volume_m3 * 1.0E6  ! 1e6 g/m^3 = 1 g/cm^3 × 1e6 cm^3/m^3
        
        ! Apply appropriate regime formula based on diameter thresholds
        IF (d_um < 134.43) THEN
            ! Regime 1
            v_t_cm_s = alphas(1) * (mass_g ** betas(1))
        ELSE IF (d_um < 1511.54) THEN
            ! Regime 2
            v_t_cm_s = alphas(2) * (mass_g ** betas(2))
        ELSE IF (d_um < 3477.84) THEN
            ! Regime 3
            v_t_cm_s = alphas(3) * (mass_g ** betas(3))
        ELSE
            ! Regime 4
            v_t_cm_s = alphas(4) * (mass_g ** betas(4))
        END IF
        
        ! Convert from cm/s to m/s
        terminal_velocity = v_t_cm_s * 1.0E-2
        
    END FUNCTION simmel_velocity

    ! Calculate surface tension using Vargaftik+ (1983) formula
    PURE ELEMENTAL FUNCTION calc_surface_tension(T) RESULT(sigma)
        IMPLICIT NONE
        real(kind(0d0)), INTENT(IN) :: T
        real(kind(0d0)) :: sigma
        real(kind(0d0)), PARAMETER :: B = 235.0E-3     ! [N/m]
        real(kind(0d0)), PARAMETER :: b2 = -0.625       ! []
        real(kind(0d0)), PARAMETER :: mu = 1.256       ! []
        real(kind(0d0)), PARAMETER :: Tc = 647.15      ! [K]
        real(kind(0d0)) :: term1, term2
        
        term1 = (Tc - T) / Tc
        term2 = 1.0 + b2 * term1
        sigma = B * (term1**mu) * term2
    END FUNCTION calc_surface_tension

    ! Calculate dynamic viscosity 
    PURE ELEMENTAL FUNCTION calc_dynamic_viscosity(T) RESULT(viscosity)
        IMPLICIT NONE
        real(kind(0d0)), INTENT(IN) :: T
        real(kind(0d0)) :: viscosity
        
        viscosity = 1.72E-5 * (393.0 / (T + 120.0)) * ((T / 273.0) ** 1.5)
    END FUNCTION calc_dynamic_viscosity

    ! Beard terminal velocity function
    FUNCTION beard_terminal_velocity(diameter, temp_air, pressure_air) RESULT(terminal_velocity)
        IMPLICIT NONE
        real(kind(0d0)), INTENT(IN) :: diameter, temp_air, pressure_air
        real(kind(0d0)) :: terminal_velocity
        
        ! Constants
        real(kind(0d0)), PARAMETER :: g = 9.81                    ! m/s^2
        real(kind(0d0)), PARAMETER :: R_air = 287.05              ! J / kg / K
        real(kind(0d0)), PARAMETER :: density_liquid = 998.0      ! kg / m^3
        real(kind(0d0)), PARAMETER :: pi = 3.14159265359
        
        ! Regime 2 beta coefficients
        real(kind(0d0)), PARAMETER :: beta1(7) = (/ &
            -0.318657E+1, &
            +0.992696, &
            -0.153193E-2, &
            -0.987059E-3, &
            -0.578878E-3, &
            +0.855176E-4, &
            -0.327815E-5 /)
            
        ! Regime 3 beta coefficients  
        real(kind(0d0)), PARAMETER :: beta2(6) = (/ &
            -0.500015E+1, &
            +0.523778E+1, &
            -0.204914E+1, &
            +0.475294, &
            -0.542819E-1, &
            +0.238449E-2 /)
        
        ! Local variables
        real(kind(0d0)) :: density_air, density_difference
        real(kind(0d0)) :: dynamic_viscosity, surface_tension
        real(kind(0d0)) :: C_1, C_2, C_3, N_da, N_p, N_re, Bo
        real(kind(0d0)) :: X, Y
        INTEGER :: k
        
        ! Check diameter range
        IF (diameter < 5.0E-7 .OR. diameter > 7.5E-3) THEN
            terminal_velocity = -999.0  ! Error value
            RETURN
        END IF
        
        ! Calculate air density
        density_air = pressure_air / (temp_air * R_air)
        density_difference = density_liquid - density_air
        
        IF (density_difference <= 0.0) THEN
            terminal_velocity = -999.0  ! Error value
            RETURN
        END IF
        
        ! Calculate helper quantities
        dynamic_viscosity = calc_dynamic_viscosity(temp_air)
        surface_tension = calc_surface_tension(temp_air)
        
        ! Determine regime and calculate terminal velocity
        IF (diameter < 19.0E-6) THEN
            ! Regime 1: Small droplets (stokes flow)
            C_1 = density_difference * g / (18.0 * dynamic_viscosity)
            terminal_velocity = C_1 * (diameter**2)
            
        ELSE IF (diameter < 1.07E-3) THEN
            ! Regime 2: Intermediate droplets
            C_2 = 4.0 * density_air * density_difference * g / (3.0 * dynamic_viscosity**2)
            N_da = C_2 * (diameter**3)
            X = LOG(N_da)
            
            ! Calculate polynomial
            Y = 0.0
            DO k = 1, 7
                Y = Y + beta1(k) * (X**(k-1))
            END DO
            
            N_re = EXP(Y)
            terminal_velocity = dynamic_viscosity * N_re / (density_air * diameter)
            
        ELSE
            ! Regime 3: Large droplets
            N_p = surface_tension**3 * density_air**2 / &
                  (dynamic_viscosity**4 * density_difference * g)
            C_3 = 4.0 * density_difference * g / (3.0 * surface_tension)
            Bo = C_3 * diameter**2
            X = LOG(Bo * (N_p**(1.0/6.0)))
            
            ! Calculate polynomial
            Y = 0.0
            DO k = 1, 6
                Y = Y + beta2(k) * (X**(k-1))
            END DO
            
            N_re = (N_p**(1.0/6.0)) * EXP(Y)
            terminal_velocity = dynamic_viscosity * N_re / (density_air * diameter)
        END IF
        
    END FUNCTION beard_terminal_velocity

END PROGRAM load_sample_data